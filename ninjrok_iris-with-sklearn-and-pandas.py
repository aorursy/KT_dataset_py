# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

% matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Iris.csv')
data.head(n=5)
data['Species'].unique()
n_records = data.shape[0]



print('Number of records: {}'.format(n_records))
data.describe()
species_raw = data['Species']

features_raw = data.drop(['Id', 'Species'], axis=1)

feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalWidthCm', 'PetalLengthCm']
def visualize(data, features):

    fig = plt.figure(figsize = (15,10));

    

    for i, feature in enumerate(features):

        ax = fig.add_subplot(2, 2, i+1)

        ax.hist(data[feature], bins = 25, color = '#00A0A0')

        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)

        ax.set_xlabel("Value")

        ax.set_ylabel("Number of Records")
visualize(data, feature_cols)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

features_df = pd.DataFrame(data=features_raw)

features_df[feature_cols] = scaler.fit_transform(features_df[feature_cols])

features_df.describe()
visualize(features_df, feature_cols)
species_df = pd.DataFrame(species_raw)

species_df['Species'] = species_df['Species'].astype('category')

species_df = species_df.apply(lambda x: x.cat.codes)

species_df.hist()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(features_df.values, species_df.values, 

                                                    train_size=0.7, test_size=0.3)
print('Sizes of:-\ntrain data-set: {}\ntest data-set: {}'.format(X_train.shape, X_test.shape))
from sklearn.metrics import fbeta_score, accuracy_score



def train_predict(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    

    predictions_train = model.predict(X_train)

    predictions_test = model.predict(X_test)

    

    results = {}

    

    results['acc_train'] = accuracy_score(y_train, predictions_train)

    results['acc_test'] = accuracy_score(y_test, predictions_test)

    results['f_train'] = fbeta_score(y_train, predictions_train, beta=1, average='micro')

    results['f_test'] = fbeta_score(y_test, predictions_test, beta=1, average='micro')

    

    return results
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier



clf_A = RandomForestClassifier(random_state=42)

clf_B = LogisticRegression(random_state=42)

clf_C = DecisionTreeClassifier(random_state=42)



results = {}



for clf in [clf_A, clf_B, clf_C]:

    results[clf.__class__.__name__] = train_predict(clf, X_train, y_train, X_test, y_test)

    

print(results)