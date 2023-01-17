# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

train_targets_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

train_targets_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")

sample_submission = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")
train_features.head()
train_targets_scored.head()
import seaborn as sns

import matplotlib.pyplot as plt



description = train_features.drop(columns=['cp_time']).describe().T.reset_index()
plt.figure(figsize=(25,8))

ax = sns.scatterplot(x="index", y="mean", data=description)

N = len(description)

ax.set_xticks(np.arange(0, N, 9))

ax.set_xticklabels(description['index'].values[::9], rotation=90, fontsize=14)

ax.set_xlabel('index', fontsize=16)

ax.set_ylabel('mean', fontsize=16);
plt.figure(figsize=(25,8))

ax = sns.scatterplot(x="index", y="std", data=description)

N = len(description)

ax.set_xticks(np.arange(0, N, 9))

ax.set_xticklabels(description['index'].values[::9], rotation=90, fontsize=14)

ax.set_xlabel('index', fontsize=16)

ax.set_ylabel('std', fontsize=16);
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import BayesianRidge

from sklearn.base import clone

from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm



class MultitargetEnsemble():

    def __init__(self,

                 categorical_columns=["cp_type", "cp_time", "cp_dose"],

                 drop_columns=['sig_id'],

                 base_model_g=ElasticNet(), 

                 base_model_c=BayesianRidge(),

                 voting_strategy="mean",

                 clip_to_zero_low_level=0):

        

        self.categorical_columns = categorical_columns

        self.drop_columns=drop_columns

        self.base_model_g = base_model_g

        self.base_model_c = base_model_c

        self.clip_to_zero_low_level = clip_to_zero_low_level

        

        if voting_strategy == 'min':

            self.voting_strategy = np.min

        elif voting_strategy == 'max':

            self.voting_strategy = np.max

        elif voting_strategy == 'mean':

            self.voting_strategy = np.mean

        else:

            raise NotImplementedError

        

    def _preprocess_features(self, features, sort=False):

        train_categorical = features[self.categorical_columns]

        train_features = features.drop(columns=self.categorical_columns)

        if sort:

            train_features = train_features.sort_values(by='sig_id')

        train_features = train_features.drop(columns=['sig_id'])

        

        features_columns = train_features.columns

        

        g_columns = [col for col in features_columns if 'g' in col]

        self.g_columns = g_columns

        

        c_columns = [col for col in features_columns if 'c' in col]

        self.c_columns = c_columns

        

        



        train_g = train_features[g_columns]

        train_c = train_features[c_columns]

        

        ohe = OneHotEncoder(categories='auto')

        categorical_OHE = ohe.fit_transform(train_categorical).toarray()

        

        TRAIN_g = pd.concat([train_g, pd.DataFrame(categorical_OHE)], axis=1)

        TRAIN_c = pd.concat([train_c, pd.DataFrame(categorical_OHE)], axis=1)

        return TRAIN_g, TRAIN_c



    def fit(self, features, target, metric=mean_squared_error):

        TRAIN_g, TRAIN_c = self._preprocess_features(features, sort=True)

        TRAIN_g, TRAIN_c = np.array(TRAIN_g), np.array(TRAIN_c)

        target = target.sort_values(by='sig_id').drop(columns=['sig_id'])

        

        self.g_models = []

        self.c_models = []

        self.target_columns = target.columns

        clip_target = lambda x: np.clip(x, 0., 1.)

        metrics_history = {'Columns':target.columns, 'G model':[], "C model":[], "Total":[]}

        for col in tqdm(target.columns):

            y = np.array(target[col])

            g_estimator_i = clone(self.base_model_g)

            g_estimator_i.fit(TRAIN_g, y)

            g_prediction = g_estimator_i.predict(TRAIN_g)

            

            self.g_models.append(g_estimator_i)

            

            c_estimator_i = clone(self.base_model_c)

            c_estimator_i.fit(TRAIN_c, y)

            c_prediction = c_estimator_i.predict(TRAIN_c)

            self.c_models.append(c_estimator_i)



            total_prediction = self.voting_strategy(np.vstack([g_prediction, c_prediction]), axis=0)

            metrics_history['G model'].append(metric(y, clip_target(g_prediction)))

            metrics_history['C model'].append(metric(y, clip_target(c_prediction)))

            total_prediction = clip_target(total_prediction)

            total_prediction[total_prediction < self.clip_to_zero_low_level] = 0.

            metrics_history['Total'].append(metric(y, total_prediction))

        metrics_history = pd.DataFrame.from_dict(metrics_history)

        return self, metrics_history

    

    def predict(self, features, hack_columns=None):

        

        prediction = {'sig_id': features['sig_id'].values}

        features_g, features_c = self._preprocess_features(features)

        

        features_g, features_c = np.array(features_g), np.array(features_c)

        for i in range(len(self.g_models)):

            g_prediction = self.g_models[i].predict(features_g)

            

            c_prediction = self.c_models[i].predict(features_c)

            total = self.voting_strategy(np.vstack([g_prediction, c_prediction]), axis=0) 

            total[total < self.clip_to_zero_low_level] = 0.

            prediction[self.target_columns[i]] = total

        return pd.DataFrame.from_dict(prediction)
model = MultitargetEnsemble()

_, train_history = model.fit(train_features, train_targets_scored)
import seaborn as sns



def plot_metric(train_history, model="G model", top_k=10):

    plt.figure(figsize=(25,7))

    indexes = np.argsort(train_history[model])[::-1]

    plt.bar(train_history['Columns'][indexes], train_history[model][indexes])

    N = len(train_history['Columns'])

    plt.xticks(np.arange(0, N, 2), train_history['Columns'][indexes][::2], rotation=90, fontsize=12)

    Text = f'Top{top_k}:\n'

    for col in train_history['Columns'][indexes][:top_k].values:

        Text += f"{col}\n"

    plt.text(100, 0.01, Text)

    plt.ylabel("MSE on target", fontsize=14)

    plt.xlabel("target", fontsize=14);
plot_metric(train_history)
plot_metric(train_history, "C model")
plot_metric(train_history, "Total")
submission = model.predict(test_features)

submission.head()
submission.to_csv("submission.csv", index=None)