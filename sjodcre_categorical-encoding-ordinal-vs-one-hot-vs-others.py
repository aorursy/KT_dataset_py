import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC

from sklearn.model_selection import KFold

import xgboost as xgb



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls





# Load dataset.

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



PassengerId = test['PassengerId']



#fill NaN values in the age column with the median of that column

train['Age'].fillna(train['Age'].mean(), inplace = True)

#fill test with the train mean to test

test['Age'].fillna(train['Age'].mean(), inplace = True)



#fill NaN values in the embarked column with the mode of that column

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)

#fill test NaN values in the embarked column with the mode from the train set

test['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)



#fill NaN values in the fare column with the median of that column

train['Fare'].fillna(train['Fare'].median(), inplace = True)

test['Fare'].fillna(train['Fare'].median(), inplace = True)



#delete the cabin feature/column and others 

drop_column = ['PassengerId','Cabin', 'Ticket']

train.drop(drop_column, axis=1, inplace = True)

test.drop(drop_column, axis=1, inplace = True)



#create a new column which is the combination of the sibsp and parch column

train['FamilySize'] = train ['SibSp'] + train['Parch'] + 1

test['FamilySize'] = test ['SibSp'] + test['Parch'] + 1



#create a new column and initialize it with 1

train['IsAlone'] = 1 #initialize to yes/1 is alone

train['IsAlone'].loc[train['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

test['IsAlone'] = 1 #initialize to yes/1 is alone

test['IsAlone'].loc[test['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1



#quick and dirty code split title from the name column

train['Title'] = train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

test['Title'] = test['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



#Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut

#Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html

train['FareBin'] = pd.qcut(train['Fare'], 4)

test['FareBin'] = pd.qcut(train['Fare'], 4)



#alternatively, you can split them yourselves based on the bins you prefer, and you can do the same for the age too

#     #Mapping Fare

#     dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

#     dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

#     dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

#     dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

#     # Mapping Age

#     dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

#     dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

#     dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

#     dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

#     dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;



#Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html

train['AgeBin'] = pd.cut(train['Age'].astype(int), 5)

test['AgeBin'] = pd.cut(train['Age'].astype(int), 5)



#so create stat_min and any titles less than 10 will be put into Misc category

stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/

title_names = (train['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

title_names_test = (test['Title'].value_counts() < stat_min)



#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/

train['Title'] = train['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

test['Title'] = test['Title'].apply(lambda x: 'Misc' if title_names_test.loc[x] == True else x)
train.tail()
#This is to show the codes for before and after

train_enc1 = train.copy()

train_enc2 = train.copy()

train_enc3 = train.copy()

train_enc4 = train.copy()

train_enc5 = train.copy()

train_enc1.head()

# train_enc2.info()
#There are a few ways to do this, so I will demonstrate for each of them.

from sklearn.preprocessing import LabelEncoder



#1st method

label = LabelEncoder()  

train_enc1['Sex_Code'] = label.fit_transform(train_enc1['Sex'])

#train_enc1.head() #now look at the Sex_code column that is created



#2nd method

train_enc1['Sex'].replace(['male','female'],[0,1],inplace=True) #the Sex column is replaced.

train_enc1.head() #now look at the sex column

# train['Embarked_Code'] = label.fit_transform(train['Embarked'])



#of course there might be other methods that do such similar things, you are free to choose :)
#There are a few ways to do this, so I will demonstrate for each of them.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import category_encoders as ce

from sklearn.compose import ColumnTransformer



#1st method

oneHot = OneHotEncoder(handle_unknown='ignore')

ce_ohe = ce.OneHotEncoder(cols = ['Sex'])

train_enc2= ce_ohe.fit_transform(train_enc2)

print(ce_ohe)#here you can will the Sex_1 and Sex_2 replaces the original Sex column

#based on the doc, it seems there isn't any param to handle the removing one column to prevent the dummy variable trap, so you might need to remove it manually



#2nd method

# ct = ColumnTransformer(

#     [('oh_enc', OneHotEncoder(sparse=False), [1]),],  # the column numbers I want to apply this to

#     remainder='passthrough'  # This leaves the rest of my columns in place

# )

# ct2 = ct.fit_transform(train_enc2)

# # df = pd.DataFrame(ct2)

# # df.head()

# print(ct.get_feature_names)

#using columntransformer will lose the column names, Idk if they will change it in the near future or not.

#you can use the get_feature names to get the column names, but still needs extra steps, refer to the link below:

#https://stackoverflow.com/questions/54646709/sklearn-pipeline-get-feature-name-after-onehotencode-in-columntransformer



#2.1 method: using sklearn's OneHotEncoder is slightly more complicated, you can refer more at the link below:

#https://stackoverflow.com/questions/43588679/issue-with-onehotencoder-for-categorical-features

#IMO, 2nd and 2.1 method is not that friendly, so why bother with extra headaches when the others seem to be able to do the same thing?



#3rd method

train_enc2 = pd.get_dummies(train_enc2, columns=['Embarked'])

train_enc2.head()#as you can see 3 new columns are created (Embarked_C, Embarked_Q and so on),you can use the drop_first param to remove one column to prevent the Dummy Variable trap



#of course there might be other methods that do such similar things, you are free to choose :)
import category_encoders as ce

ce_bin = ce.BinaryEncoder(cols = ['Embarked'])

train_enc3 = ce_bin.fit_transform(train_enc3)

train_enc3.head()
import category_encoders as ce

ce_bin = ce.BaseNEncoder(cols = ['Embarked'])

train_enc4 = ce_bin.fit_transform(train_enc4)

train_enc4.head()
import category_encoders as ce

ce_bin = ce.HashingEncoder(cols = ['Embarked'])

train_enc5 = ce_bin.fit_transform(train_enc5)

train_enc5.head()