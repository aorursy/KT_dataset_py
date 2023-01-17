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


# Data visualization

import seaborn as sns

import matplotlib.pyplot as plt



# Scalers

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle



# Models

from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn.linear_model import Perceptron

from sklearn import svm #support vector Machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.neural_network import MLPClassifier



# Cross-validation

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.model_selection import cross_validate



# GridSearchCV

from sklearn.model_selection import GridSearchCV



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.plotting import scatter_matrix
train_df = pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")

#Combine Both the datasets

data_df = train_df.append(test_df)
data_df['Title'] = data_df['Name']

# Cleaning name and extracting Title

for name_string in data_df['Name']:

    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)
# Replacing rare titles with more common ones

mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

data_df.replace({'Title': mapping}, inplace=True)

titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']



#Calculate the Median age for each Title to fill the missing age

for title in titles:

    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]

    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute



# Substituting Age values in TRAIN_DF and TEST_DF. We have 891 because we know the first 891 entries are Train and remaining are Test data

train_df['Age'] = data_df['Age'][:891]

test_df['Age'] = data_df['Age'][891:]
# Dropping Title feature

data_df.drop('Title', axis = 1, inplace = True)
#Adding Family_Size i.e Parch (Parent childer) + SibSp (Sibbling). 

#In Other words we are trying to find out if they have any family or if they travel alone.

data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']



# Substituting Family Size in TRAIN_DF and TEST_DF:

train_df['Family_Size'] = data_df['Family_Size'][:891]

test_df['Family_Size'] = data_df['Family_Size'][891:]
#Adding a new Feature Family suriving detail.



# Identify the family based on the Lastname and Fare.

data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])

data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)



#

DEFAULT_SURVIVAL_VALUE = 0.5

data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE



for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    

    if (len(grp_df) != 1):

        # A Family group is found.

        for ind, row in grp_df.iterrows():

            smax = grp_df.drop(ind)['Survived'].max()

            smin = grp_df.drop(ind)['Survived'].min()

            passID = row['PassengerId']

            if (smax == 1.0):

                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1

            elif (smin==0.0):

                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0



print("Number of passengers with family survival information:", 

      data_df.loc[data_df['Family_Survival']!=0.5].shape[0])
for _, grp_df in data_df.groupby('Ticket'):

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin==0.0):

                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

                        

print("Number of passenger with family/group survival information: " 

      +str(data_df[data_df['Family_Survival']!=0.5].shape[0]))



# # Family_Survival in TRAIN_DF and TEST_DF:

train_df['Family_Survival'] = data_df['Family_Survival'][:891]

test_df['Family_Survival'] = data_df['Family_Survival'][891:]
#Lets Handle the Fare_bin now

data_df['Fare'].fillna(data_df['Fare'].median(), inplace = True)



# Making Bins

data_df['FareBin'] = pd.qcut(data_df['Fare'], 5)



label = LabelEncoder()

data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])



train_df['FareBin_Code'] = data_df['FareBin_Code'][:891]

test_df['FareBin_Code'] = data_df['FareBin_Code'][891:]



train_df.drop(['Fare'], 1, inplace=True)

test_df.drop(['Fare'], 1, inplace=True)
#Making Age_Bins

data_df['AgeBin'] = pd.qcut(data_df['Age'], 4)



label = LabelEncoder()

data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])



train_df['AgeBin_Code'] = data_df['AgeBin_Code'][:891]

test_df['AgeBin_Code'] = data_df['AgeBin_Code'][891:]



train_df.drop(['Age'], 1, inplace=True)

test_df.drop(['Age'], 1, inplace=True)
#Mapping the Gender to number 

train_df['Sex'].replace(['male','female'],[0,1],inplace=True)

test_df['Sex'].replace(['male','female'],[0,1],inplace=True)
#Creating the back up of my data before Dropping. 

#I get paranoid whenever I drop columns so, just making sure I dont lose anything

train_backup=train_df.copy()

test_backup=test_df.copy()
#Drop the unwanted Columns

train_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',

               'Embarked'], axis = 1, inplace = True)

test_df.drop(['Name','PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',

              'Embarked'], axis = 1, inplace = True)
X = train_df.drop('Survived', 1)

y = train_df['Survived']

X_test = test_df.copy()
#Scaling features

std_scaler = StandardScaler()

X = std_scaler.fit_transform(X)

X_test = std_scaler.transform(X_test)
#Grid Search CV. We are using KNN

n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]

algorithm = ['auto']

weights = ['uniform', 'distance']

leaf_size = list(range(1,50,5))

hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 

               'n_neighbors': n_neighbors}

gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 

                cv=10, scoring = "roc_auc")

gd.fit(X, y)

print(gd.best_score_)

print(gd.best_estimator_)
#Using a model found by grid searching

gd.best_estimator_.fit(X, y)

y_pred = gd.best_estimator_.predict(X_test)
output3 = pd.DataFrame(pd.read_csv("../input/titanic/test.csv")['PassengerId'])

output3['Survived'] = y_pred

output3.to_csv("mysubmission3.csv", index = False)