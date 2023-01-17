import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np



%matplotlib inline
df = pd.read_csv('../input/phishing-website-dataset/dataset.csv')
# First 5 sample rows of Dataset

df.head()
# Name of columns

list(df.columns)
df.info()
for col in df.columns:

    unique_value_list = df[col].unique()

    if len(unique_value_list) > 10:

        print(f'{col} has {df[col].nunique()} unique values')

    else:

        print(f'{col} contains:\t\t\t{unique_value_list}')
df = df.drop(columns=['index'])
print(df['Result'].value_counts())

sns.countplot(df['Result'])
plt.figure(figsize=(15, 15))

sns.heatmap(df.corr(), linewidths=.5)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score





def binary_classification_accuracy(actual, pred):

    

    print(f'Confusion matrix: \n{confusion_matrix(actual, pred)}')

    print(f'Accuracy score: \n{accuracy_score(actual, pred)}')

    print(f'Classification report: \n{classification_report(actual, pred)}')
# Replacing -1 with 0 in the target variable

df['Result'] = np.where(df['Result']==-1, 0, df['Result'])

target = df['Result']

features = df.drop(columns=['Result'])
folds = KFold(n_splits=4, shuffle=True, random_state=42)



train_index_list = list()

validation_index_list = list()



for fold, (train_idx, validation_idx) in enumerate(folds.split(features, target)):

    

    model = XGBClassifier()

    model.fit(np.array(features)[train_idx,:], np.array(target)[train_idx])

    predicted_values = model.predict(np.array(features)[validation_idx,:])

    print(f'==== FOLD {fold+1} ====')

    binary_classification_accuracy(np.array(target)[validation_idx], predicted_values)