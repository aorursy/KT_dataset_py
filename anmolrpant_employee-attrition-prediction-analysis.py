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
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler

training_set = pd.read_csv("/kaggle/input/summeranalytics2020/train.csv")

testing_set = pd.read_csv("/kaggle/input/summeranalytics2020/test.csv")

sample = pd.read_csv("/kaggle/input/summeranalytics2020/Sample_submission.csv")

training_set.head()
training_set.describe()
#Checking Missing Values

cols_with_missing = [col for col in training_set.columns if training_set[col].isnull().any()]

print(cols_with_missing)
training_set.columns
#Checking for Categorical_Variables

s = (training_set.dtypes == 'object')

object_cols = list(s[s].index)

print("Categorical variables:")

print(object_cols)
#Data Visualization

plt.figure(figsize=(100,75))

cor = training_set.corr()

sns.heatmap(cor,annot=True,annot_kws={"size": 50})
from sklearn.preprocessing import LabelEncoder

for col in training_set.columns:

        if (training_set[col].dtypes == 'object'):

            training_set[col] = LabelEncoder().fit_transform(training_set[col])

for col in testing_set.columns:

        if (testing_set[col].dtypes == 'object'):

            testing_set[col] = LabelEncoder().fit_transform(testing_set[col])
X_train = training_set.drop(['Attrition','Id','Behaviour'],axis=1)

Y_train = training_set.Attrition

X_test = testing_set.drop(['Id','Behaviour'],axis=1)
#from sklearn.feature_selection import SelectKBest

#from sklearn.feature_selection import chi2

#sel = SelectKBest(chi2,k=21)

#sel.fit_transform(X_train,Y_train)

#mask = sel.get_support() #list of booleans

#new_features = [] # The list of your K best features

#feature_names = list(X_train.columns.values)

#for bool, feature in zip(mask, feature_names):

#    if bool:

#        new_features.append(feature)

#print(new_features)
#X_train = X_train[new_features]

#X_test = X_test[new_features]
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(activation='relu')

clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)

pprob=clf.predict_proba(X_test)

ss=pprob[:,1]

sample=sample.drop('Attrition',axis=1)

sample['Attrition']=ss

sample.to_csv('submission.csv',index=False) 