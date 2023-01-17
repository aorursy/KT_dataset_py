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

# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



#Evaluation

from sklearn.metrics import accuracy_score, classification_report
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')

df_sub   = pd.read_csv('../input/gender_submission.csv')
df_train.head()
df_all = [df_train, df_test] #New list with both Train and Test data as it's elements

#train -> df_all[0]

#test  -> df_all[1] 
df_all[0][:5]
df_all[1][:5] #or .head()
print(df_train.columns.values)  # or use tolist() to convert ir into a list from Index Array
def Nan_data(df):

    key = df.isnull().sum().index.values

    value = df.isnull().sum().values

    return dict(zip(key,value))
Nan_dict_train = Nan_data(df_train)

Nan_dict_test  = Nan_data(df_test)
print('Training data Nan summary: \n {} \n'.format(Nan_dict_train))

print('Test data Nan summary: \n {}'.format(Nan_dict_test))
#print(tuple(zip(df_train.dtypes.index,df_train.dtypes.values)))

df_train.dtypes
Survived = df_train[df_train.Survived == 1].Survived.count()

print('No. of people survived in the Train dataset: {}'.format(Survived))
Died = df_train[df_train.Survived == 0].Survived.count()

print('No. of people died in the Train dataset: {}'.format(Died))
total = Survived+Died

survival_rate = Survived/total

print('Survived: {} \nSurvival_rate: {:.2f}%'.format(Survived, survival_rate*100))
df_train.describe(percentiles=[0.25,0.50,0.61,0.62,0.75])
print(tuple(zip(df_train.dtypes.index,df_train.dtypes.values)))
Cat_col = ([x[0] for x in tuple(zip(df_train.dtypes.index,df_train.dtypes.values)) 

           if x[1] == np.dtype('O') and x[0] not in ['Name','Cabin','Ticket']])

#df_train.Sex.dtype = dtype('O')

Cat_col
#df_train[Cat_col].describe() 

df_train.describe(include=['O']) 
df_train.corr()
def Feat_en(df):

    df['Family_Size'] = df.Parch + df.SibSp + 1

    df['SoloTravel'] = (df['Family_Size'] <= 1)

    df.loc[df['SoloTravel']==False,'SoloTravel'] = 0

    df.loc[df['SoloTravel']==True,'SoloTravel'] = 1

    # df[df['Family_Size'] <= 1, 'SoloTravel'] = 1

    # df[df['Family_Size'] > 1, 'SoloTravel'] = 0

    return df
for df in df_all:

    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)
# fillinf the Nan values for Embarked and Fare columns in Train and Test dataset respectively.

df_train['Embarked'].fillna(df_train['Embarked'].dropna().mode()[0], inplace=True)

df_test['Fare'].fillna(df_train['Fare'].dropna().mean(), inplace=True)
for df in df_all:

    df['Embarked'] = df['Embarked'].map({'C': 0, 'S': 1, 'Q': 2}).astype(int)
df_all[0].head()
guess_ages = np.zeros((2,3))



for df in df_all:

    for i in range(0, 2):

        for j in range(1, 4):

            df_age = df[(df['Sex'] == i) & (df['Pclass'] == j)]['Age'].dropna()

            age_guess = df_age.median()

            # Convert random age float to nearest .5 age

            guess_ages[i,j-1] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(1, 4):

            df.loc[(df.Age.isnull() & (df.Sex==i) & (df.Pclass==j)), 'Age'] = guess_ages[i,j-1]



    df['Age'] = df['Age'].astype(int)
df_test.isnull().sum()
df_train.isnull().sum()
# Take copy of original train/test dataframe before aplpying Feat_engineering

df_train_0 = df_train.copy() 

df_test_0 = df_test.copy()

#Apply Feat_en

train_df = Feat_en(df_train)

test_df = Feat_en(df_test)
#check if there are nay nan/Null values in the dataframe

print(train_df.isnull().sum())

print(test_df.isnull().sum())
train_df.head()
def drop_col(df_todrop):

    drop_col = ['PassengerId', 'SibSp', 'Parch', 'Family_Size', 'Name', 'Cabin','Ticket']

    rem_col = [col for col in drop_col if col in df_todrop.columns.tolist()]

    return df_todrop.drop(columns=rem_col)
train_df = drop_col(train_df)

test_df = drop_col(test_df)
train_df.head(3)
test_df.head(3)
train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train_df[['SoloTravel','Survived']].groupby(['SoloTravel'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)
g=sns.FacetGrid(train_df,col='Survived')

g.map(plt.hist, 'Pclass', bins=15)
g=sns.FacetGrid(train_df,col='Survived')

g.map(plt.hist, 'Age', bins=15)
train_df[(train_df['Age'] >= 20) & (train_df['Age'] <= 40)].count()[0]
print('Percentage of passengers in 20-40 age group: {:.2f}%'.format((579/891)*100))
#Count of people who Died in 20-40 Age group

train_df[(train_df['Age'] >= 20) & (train_df['Age'] <= 40) & (train_df['Survived'] == 0)].count()[0]
#Count of people who Survived in 20-40 Age group

train_df[(train_df['Age'] >= 20) & (train_df['Age'] <= 40) & (train_df['Survived'] == 1)].count()[0]
print('Percentage of passengers in 20-40 age group who survived: {:.2f}%'.format((208/579)*100))
g=sns.FacetGrid(train_df,col='Survived')

g.map(plt.hist, 'SoloTravel', bins=15)
g=sns.FacetGrid(train_df,col='Survived')

g.map(plt.hist, 'Fare', bins=15)
g=sns.FacetGrid(train_df,col='Survived')

g.map(plt.hist, 'Embarked', bins=15)
sns.catplot(x='Pclass',y='Survived',kind='bar',hue='Sex', data=train_df) 

# hue distributes the plot as per provided col. Here, Sex
sns.catplot(x='Survived',y='Fare',kind='bar',hue='Sex',data=train_df)
sns.catplot(x='Pclass',hue='Sex',kind='count',data=train_df)
train_df.head()
def get_OHE(df_toOHE):

    cat_cols = ['Pclass','Sex','Embarked','SoloTravel']

    df_dummies = pd.DataFrame()

    for col in cat_cols:

        df_dummies = pd.concat([df_dummies,pd.get_dummies(df_toOHE[col], prefix=col)],axis=1)

    return df_dummies

    

train_dummies = get_OHE(train_df)

test_dummies = get_OHE(test_df)
train_dummies.head(2)
def Merge_OHE(df_toMOHE, df_OHE):

    drop_col = ['Pclass','Sex','Embarked','SoloTravel']

    return df_toMOHE.merge(df_OHE, left_index=True, right_index=True).drop(columns=drop_col)
df_train_1 = Merge_OHE(train_df, train_dummies)

df_test_1 = Merge_OHE(test_df, test_dummies)
df_train_1.iloc[:,1:].head()
X = df_train_1.iloc[:,1:]

y = df_train_1.iloc[:,0]



scaler = MinMaxScaler()  

#MinMAxScaling doesn't work on Categorical features as they are alreary scaled between 0 & 1.

#X[['Age','Fare']] = scaler.fit_transform(X[['Age','Fare']])

#X_df = pd.DataFrame(scaler.fit_transform(X), columns=df_train_1.columns[1:])

#df_test_1[['Age','Fare']] = scaler.transform(df_test_1[['Age','Fare']])

#X_test_df = pd.DataFrame(scaler.transform(df_test_1), columns=df_test_1.columns)



X = scaler.fit_transform(X)

X_sub = scaler.transform(df_test_1)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 56, test_size = 0.25)
model_0 = DecisionTreeClassifier(random_state=56)

model_0.fit(X_train,y_train)

print('Train Set Accuracy: {:.2f}%'.format(model_0.score(X_train,y_train)*100))

print('Test Set Accuracy: {:.2f}%'.format(model_0.score(X_test,y_test)*100))
model_1 = RandomForestClassifier(n_estimators=220, random_state=56)

model_1.fit(X_train,y_train)

print('Train Set Accuracy: {:.2f}%'.format(model_1.score(X_train,y_train)*100))

print('Test Set Accuracy: {:.2f}%'.format(model_1.score(X_test,y_test)*100))
model_2 = LogisticRegression(random_state=56)

model_2.fit(X_train,y_train)

print('Train Set Accuracy: {:.2f}%'.format(model_2.score(X_train,y_train)*100))

print('Test Set Accuracy: {:.2f}%'.format(model_2.score(X_test,y_test)*100))
model_3 = SVC(kernel = 'linear',gamma='auto',random_state=56)

model_3.fit(X_train,y_train)

print('Train Set Accuracy: {:.2f}%'.format(model_3.score(X_train,y_train)*100))

print('Test Set Accuracy: {:.2f}%'.format(model_3.score(X_test,y_test)*100))
model_4 = SVC(kernel = 'rbf', gamma='auto')

model_4.fit(X_train,y_train)

print('Train Set Accuracy: {:.2f}%'.format(model_4.score(X_train,y_train)*100))

print('Test Set Accuracy: {:.2f}%'.format(model_4.score(X_test,y_test)*100))
for k in range(4,15):

    model_5 = KNeighborsClassifier(n_neighbors=k)

    model_5.fit(X_train,y_train)

    print('k={} Train Set Accuracy: {:.2f}%'.format(k,model_5.score(X_train,y_train)*100))

    print('k={} Test Set Accuracy: {:.2f}%\n'.format(k,model_5.score(X_test,y_test)*100))
k=12

model_5 = KNeighborsClassifier(n_neighbors=k)

model_5.fit(X_train,y_train)

print('k={} Train Set Accuracy: {:.2f}%'.format(k,model_5.score(X_train,y_train)*100))

print('k={} Test Set Accuracy: {:.2f}%'.format(k,model_5.score(X_test,y_test)*100))
Y_pred = model_3.predict(X_sub)
df_sub.Survived = Y_pred

df_sub.head()
df_sub.to_csv('Sub_1.csv',index=False)
print(os.listdir("../working"))