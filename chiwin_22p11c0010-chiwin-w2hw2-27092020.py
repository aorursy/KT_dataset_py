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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support,classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
# Explore train data
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
train_df
train_df.isna().sum()
train_df['Age'] = train_df['Age'].fillna(value=train_df['Age'].mean())
train_df.isna().sum()
train_df['Cabin'].value_counts()
for i,cabin in enumerate(train_df['Cabin']):
    if type(cabin) == str:
        train_df.loc[i, 'Cabin'] = cabin[0]
train_df['Cabin'].value_counts()
train_df['Cabin'] = train_df['Cabin'].fillna(value='X')
train_df['Cabin'].value_counts()
train_df.isna().sum()
train_df['Embarked'].value_counts()
train_df['Embarked'] = train_df['Embarked'].fillna(value='S')
train_df.isna().sum()
train_df
# Convert data to indexing number
Sex_group_list = pd.unique(train_df.Sex).tolist()
Cabin_list = pd.unique(train_df.Cabin).tolist()
Embarked_list = pd.unique(train_df.Embarked).tolist()

train_df['Sex'] = pd.Categorical(train_df['Sex'],categories=Sex_group_list)
train_df['Sex'] = train_df['Sex'].cat.codes
train_df['Cabin'] = pd.Categorical(train_df['Cabin'],categories=Cabin_list)
train_df['Cabin'] = train_df['Cabin'].cat.codes
train_df['Embarked'] = pd.Categorical(train_df['Embarked'],categories=Embarked_list)
train_df['Embarked'] = train_df['Embarked'].cat.codes
train_df
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(train_df.corr(), annot=True, linewidths=0.5, fmt= '.2f')
dataset = train_df.drop(['Name','Ticket'],axis=1)
dataset
Xdata = dataset.drop(['Survived','Age','SibSp','Pclass','PassengerId'],axis=1)
Ydata = dataset['Survived']
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
scoring = {'prec_macro': 'precision_macro','rec_macro': make_scorer(recall_score, average='macro')}
DecisionTree_model = DecisionTreeClassifier(max_depth=100)
DecisionTree_cv = cross_validate(DecisionTree_model, Xdata, Ydata, cv=5,scoring=scoring,return_train_score=True)
print("Precision :",DecisionTree_cv['test_prec_macro'])
print("Recall :",DecisionTree_cv['test_rec_macro'])
F1_list = []
for i in range(5):
    #F1 = 2 * (precision * recall) / (precision + recall)
    F1 = 2* (DecisionTree_cv['test_prec_macro'][i]*DecisionTree_cv['test_rec_macro'][i])/(DecisionTree_cv['test_prec_macro'][i] +DecisionTree_cv['test_rec_macro'][i])
    F1_list.append(F1)
print("F-Measure :",F1_list)
print("F-Measure_aveage :",np.array(F1_list).mean())
GaussianNB_model = GaussianNB()
GaussianNB_cv = cross_validate(GaussianNB_model, Xdata, Ydata, cv=5,scoring=scoring,return_train_score=True)
print("Precision :",GaussianNB_cv['test_prec_macro'])
print("Recall :",GaussianNB_cv['test_rec_macro'])
F1_list = []
for i in range(5):
    #F1 = 2 * (precision * recall) / (precision + recall)
    F1 = 2* (GaussianNB_cv['test_prec_macro'][i]*GaussianNB_cv['test_rec_macro'][i])/(GaussianNB_cv['test_prec_macro'][i] +GaussianNB_cv['test_rec_macro'][i])
    F1_list.append(F1)
print("F-Measure :",F1_list)
print("F-Measure_aveage :",np.array(F1_list).mean())
from sklearn.neural_network import MLPClassifier
NN_model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)
NN_model_cv = cross_validate(NN_model, Xdata, Ydata, cv=5,scoring=scoring,return_train_score=True)
print("Precision :",NN_model_cv['test_prec_macro'])
print("Recall :",NN_model_cv['test_rec_macro'])
F1_list = []
for i in range(5):
    #F1 = 2 * (precision * recall) / (precision + recall)
    F1 = 2* (NN_model_cv['test_prec_macro'][i]*NN_model_cv['test_rec_macro'][i])/(NN_model_cv['test_prec_macro'][i] +NN_model_cv['test_rec_macro'][i])
    F1_list.append(F1)
print("F-Measure :",F1_list)
print("F-Measure_aveage :",np.array(F1_list).mean())