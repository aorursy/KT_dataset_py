# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%load_ext autoreload
%autoreload 2

%matplotlib inline
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
trainData = pd.read_csv("../input/train.csv")
testData = pd.read_csv("../input/test.csv")
trainData.head()
testData.head()
trainData.head()
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
display_all(trainData.head(20).T)
display_all(trainData.describe(include='all'))
trainData.info()
#Let us look at both training and test dataset now
def get_combined_data():
    # reading train data
    train = pd.read_csv('../input/train.csv')
    
    # reading test data
    test = pd.read_csv('../input/test.csv')

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop('Survived',1,inplace=True)
    

    # merging train data and test data for future feature engineering
    data = train.append(test)
    data.reset_index(inplace=True)
    data.drop('index',inplace=True,axis=1)
    
    return data
data = get_combined_data()
data.shape
#let us now try and get some more information
def get_titles():

    global data
    
    # we extract the title from each name
    data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    data['Title'] = data.Title.map(Title_Dictionary)
get_titles()
data.head()
def process_age():
    
    global data
    
    # a function that fills the missing values of the Age variable
    
    def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26
    
    data.Age = data.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)
    
process_age()
data.info()
data.drop('Name',axis=1,inplace=True)
data.info()
# there's one missing fare value - replacing it with the mean.
data.Fare.fillna(data.Fare.mean(),inplace=True)
data.info()
# two missing embarked values - filling them with the most frequent one (S)
data.Embarked.fillna('S',inplace=True)
data.info()
# replacing missing cabins with U (for Uknown)
data.Cabin.fillna('U',inplace=True)
data.info()
train_cats(data)
data.info()
data.Sex.cat.set_categories(['male', 'female'], ordered=True, inplace=True)
data.Sex = data.Sex.cat.codes
data.info()
display_all(data.isnull().sum().sort_index()/len(data))
data.Embarked.cat.set_categories(['C', 'Q', 'S'], ordered=True, inplace=True)
data.Embarked = data.Embarked.cat.codes
data.info()
data.Cabin = data.Cabin.cat.codes
data.info()
data.Title = data.Title.cat.codes
data.info()
data.drop('Ticket',axis=1,inplace=True)
data.info()
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
#Split data to validation and train
train = pd.read_csv('../input/train.csv')
    
    # reading test data
test = pd.read_csv('../input/test.csv')
m = len(train)
n = len(test)

x_train = data[0:m]
x_test = data[m:m + n]

train_percent = 0.80
validate_percent = 0.20


m = len(x_train)
y_train = train['Survived']

x_train = x_train[:int(train_percent * m)]
x_validation = x_train[int(validate_percent * m):]

y_train = y_train[:int(train_percent * m)]
y_validation = y_train[int(validate_percent * m):]
def get_result(predicted):
    print("F1_Score: " + str(f1_score(y_validation, predicted, average='macro')))
    print("accuracy: " + str(accuracy_score(y_validation, predicted)))
    print("AUC: " + str(roc_auc_score(y_validation, predicted)))
    print("recall: " + str(recall_score(y_validation, predicted)))
    return
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

# Random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc = rfc.fit(x_train, y_train)

print("- Random forest Validation Set -")
y_predicted = rfc.predict(x_validation)
get_result(y_predicted)
y_predicted.size

print("- Random forest Test Set-")
y_predicted = rfc.predict(x_test)
y_predicted.size
raw_data = {'PassengerId' : x_test.PassengerId, 'Survived' : y_predicted}
df = pd.DataFrame(raw_data, columns = ['PassengerId', 'Survived'])
df.to_csv('submissionFastAI2.csv', encoding='utf-8', mode = 'w', index=False)
subData = pd.read_csv('submissionFastAI2.csv')
subData.head()