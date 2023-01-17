# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import random

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import featuretools as ft



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score



from matplotlib import pyplot as plt

import seaborn as sns



from fastai import *

from fastai.tabular import *

from fastai.basic_data import DataBunch

# from tqdm import tqdm_notebook



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

%reload_ext autoreload

%autoreload 2
indir = '../input'
df = pd.read_csv(os.path.join(indir, 'train.csv'))

df_summary = df.describe(); df_summary
test_df = pd.read_csv(os.path.join(indir, 'test.csv')).set_index('ID_code')

test_df.describe()
# get the features list

features = list(test_df.columns)

len(features)
dmean, dmin, dmax = df_summary.loc['mean'],df_summary.loc['min'], df_summary.loc['max'] 

drange = dmax - dmin
df.loc[:, features] = (df[features] - dmin[features])/drange[features]

df.describe()
test_df.loc[:, features] = (test_df[features] - dmin[features])/drange[features]

test_df.describe()
def augment_df(df):

    for feature in features:

        df[f'sq_{feature}'] = df[feature]**2

        df[f'repo_{feature}'] = df[feature].apply(lambda x: 0 if x==0 else 1/x)

        df[f'repo_sq_{feature}'] = df[f'repo_{feature}']**2

        df[f'cube_{feature}'] = df[feature]**3

#         df[f'repo_cube_{feature}'] = df[f'repo_{feature}']**3

#         df[f'p4_{feature}'] = df[feature]**4

#         df[f'sin_{feature}'] = sin(df[feature])

#         df[f'exp_{feature}'] = exp(df[feature])

#         df[f'log_{feature}'] = df[f'sq_{feature}'].apply(lambda x: 0 if x==0 else log(x))

    

    df['min'] = df[features].min(axis=1)

    df['mean'] = df[features].mean(axis=1)

    df['max'] = df[features].max(axis=1)

    df['sum'] = df[features].sum(axis=1)

    df['median'] = df[features].median(axis=1)

    df['std'] = df[features].std(axis=1)

    df['var'] = df[features].var(axis=1)

    df['abs_sum'] = df[features].abs().sum(axis=1)

    df['abs_mean'] = df[features].abs().mean(axis=1)

    df['abs_median'] = df[features].abs().median(axis=1)

    df['abs_std'] = df[features].abs().std(axis=1)

    df['skew'] = df[features].skew(axis=1)

    df['kurt'] = df[features].kurt(axis=1)

    

    df['sq_kurt'] = df[[f'sq_{feature}' for feature in features]].kurt(axis=1)

    
%%time

augment_df(df)

augment_df(test_df)
features = list(test_df.columns[:-12])

stats_features = list(test_df.columns[-12:])

num_features = len(features)

num_features
### Feature Understanding
def view_dist(df, columns, row=10, col=10):

    fig, axes = plt.subplots(10,10,figsize=(30,30))

    axes = axes.flatten()

    for col, ax in zip(columns, axes):

        sns.kdeplot(df.loc[df.target==0, col], ax=ax, color='r', label='0')

        sns.kdeplot(df.loc[df.target==1, col], ax=ax, color='b', label='1')

        ax.legend()

        ax.set_title(f'{col}')

        

    plt.show()    
# view_dist(df, features[:100])
# seed = 2019

# train_samples = df.sample(frac=0.95, random_state=seed)

# valid_samples = df.drop(train_samples.index)
random.seed(31415926)

valid_idx = random.sample(list(df.index.values), int(len(df)*0.2) )

train_idx = df.drop(valid_idx).index
# verify that positive sample distribution in validation set is similar to that of the whole data

df.iloc[valid_idx].target.sum() / len(valid_idx) , df.target.sum() / len(df)
class roc(Callback):

    '''

    Updated on March 28 2019 to reflect new change in FastAI's Callback

    ROC_AUC metric callback for fastai. Compute ROC score over each batch and returns the average over batches.

    TO DO: rolling average

    '''

    def on_epoch_begin(self, **kwargs):

        self.total = 0

        self.batch_count = 0

    

    def on_batch_end(self, last_output, last_target, **kwargs):

        preds = F.softmax(last_output, dim=1)

        # roc_auc_score does not work on batches which does not contain both classes.

        try:

            roc_score = roc_auc_score(to_np(last_target), to_np(preds[:,1]))

            self.total += roc_score

            self.batch_count += 1

        except:

            pass

    

    def on_epoch_end(self, last_metrics, **kwargs):

        return add_metrics(last_metrics, self.total/self.batch_count)
BATCH_SIZE = 2048
# data = TabularDataBunch.from_df(path='.', df=df, 

#                                 dep_var='target', 

#                                 valid_idx=valid_samples.index, 

#                                 cat_names=[], 

#                                 cont_names=features, 

#                                 procs=[tabular.transform.Normalize],

#                                 test_df=test_df)



#learner = tabular_learner(data, layers=[200,100], ps=[0.5,0.2], metrics=[accuracy, roc()])



#learner.lr_find()

#learner.recorder.plot()
def train_and_eval_tabular_learner(train_df,

                                   train_features, 

                                   valid_idx,

                                   add_noise=False,

                                   lr=0.02, epochs=1, layers=[200, 50], ps=[0.5, 0.2], name='learner'):

    

    data = TabularDataBunch.from_df(path='.', df=train_df, 

                                    dep_var='target', 

                                    valid_idx=valid_idx, 

                                    cat_names=[], 

                                    cont_names=train_features, 

                                    bs=BATCH_SIZE,

                                    procs=[],

                                    test_df=test_df)

    learner = tabular_learner(data, layers=layers, ps=ps, metrics=[roc()])



    learner.fit_one_cycle(epochs, lr)



    # learner.save(name,with_opt=False)

        

    # run prediction on validation set

    valid_predicts, _ = learner.get_preds(ds_type=DatasetType.Valid)

    valid_probs = np.array(valid_predicts[:,1])

    valid_targets = train_df.loc[valid_idx].target.values

    valid_score = roc_auc_score(valid_targets, valid_probs)

    

    # run prediction on test    

    test_predicts, _ = learner.get_preds(ds_type=DatasetType.Test)

    test_probs = to_np(test_predicts[:, 1])



    return valid_score, valid_probs, test_probs
%%time

sub_features = []

valid_scores = []

valid_predictions = []

predictions = []

num_learner = 1000

num_epochs = 5

saved_model_prefix = 'learner'



for i in range(num_learner):

    print('training model {:}'.format(i))

    sub_features.append(random.sample(list(features), int(num_features*0.5)) + stats_features)

    name = f'{saved_model_prefix}_{i}'



    score, valid_probs, test_probs = train_and_eval_tabular_learner(df,

                                                                    sub_features[-1], 

                                                                    valid_idx, 

                                                                    epochs=num_epochs, 

                                                                    lr=0.01, 

                                                                    name=name)

    

    valid_scores.append(score)

    valid_predictions.append(valid_probs)

    predictions.append(test_probs)
print(valid_scores)
# roc_auc_score on validation set

average_valid_predicts = sum(valid_predictions)/len(valid_predictions)

valid_auc_score = roc_auc_score(df.iloc[valid_idx].target, average_valid_predicts); valid_auc_score
from sklearn.metrics import roc_curve



fpr, tpr, _ = roc_curve(y_true=df.iloc[valid_idx].target,y_score=average_valid_predicts)

plt.figure(figsize=(9,9))

plt.plot(fpr, tpr)

plt.show()
# this is if we want to average only on the models that score more than average

# predicts = np.zeros(predictions[0].shape)

# counts = 0

# for i in range(num_epochs):

#     if valid_scores[i] > average_valid_score:

#         predicts += predictions[i]

#         counts += 1

        

# print("number of models: {:}".format(counts))



# predicts = sum(predictions)/counts
test_df['target'] = sum(predictions)/len(valid_predictions)
# add timestamp to submission

from datetime import datetime

now = datetime.now()

model_time = now.strftime("%Y%m%d-%H%M")
test_df[['target']].to_csv(f'submission_fastai_ensemble_{model_time}_{valid_auc_score}.csv')
from IPython.display import FileLink

FileLink(f'submission_fastai_ensemble_{model_time}_{valid_auc_score}.csv')