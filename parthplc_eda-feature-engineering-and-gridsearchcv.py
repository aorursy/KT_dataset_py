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
#Importing python libraries

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

import matplotlib.pylab as pylab

import matplotlib.pyplot as plt

from pandas import get_dummies

import matplotlib as mpl

import xgboost as xgb

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib

import warnings

import sklearn

import scipy

import numpy

import json

import sys

import csv

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import Perceptron

import os

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn import linear_model
pd.set_option('display.max_rows',10)#So that we can see the whole dataset at one go
# import train and test to play with it

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
#get the type

type(train_df)


test_df.info()
train_df.info()
test_df['Survived'] = -888 #Adding Survived with a default value
test_df.info()
#test_df.head()

train_df.head()
train_df = train_df[['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Survived']]


#Concatinating two data frames(train and test)

df = pd.concat((train_df,test_df),axis = 0)



df = df.reset_index()

df.info()
df = df.drop(['index'],axis=1)



df.head()

df = df.set_index('PassengerId')

df.tail()






df.Name.head()
df.loc[5:10]
#indexing : use iloc based indexing
df.iloc[5:10,3:8]
#filter rows based on the condition

male_passengers = df.loc[df.Sex =='male',:]

print('Number of male passengers : {0}'.format(len(male_passengers)))

#use & or | operators to build complex logic
male_passengers_first_class = df.loc[((df.Sex =='male') &(df.Pclass == 1)),:]
print('Number of passengers in first class:{0}'.format(len(male_passengers_first_class)))
# use .describe() to get statistics for all numeric columns

df.describe()
df.isnull().sum()
#Numerical feature

#centrality measures
print('Mean Fare : {0}'.format(df.Fare.mean()))

print('Median Fare : {0}'.format(df.Fare.median()))

#dispersion measure

print('Max fare  : {0}'.format(df.Fare.max()))#max

print('Min fare  : {0}'.format(df.Fare.min()))#max

print('Fare range  : {0}'.format(df.Fare.max() - df.Fare.min()))#range

print('25 percentile  : {0}'.format(df.Fare.quantile(.25)))#25 percentile

print('50 percentile  : {0}'.format(df.Fare.quantile(.50)))#50 percentile

print('75 percentile  : {0}'.format(df.Fare.quantile(.75)))#75 percentile

print('Variance fare: {0}'.format(df.Fare.var()))#variance

print('Standard deviation  : {0}'.format(df.Fare.std()))#standard deviation





# box-whiskers plot

df.Fare.plot(kind='box')
#use describe to get statistics for all columns including non-numeric ones

df.describe(include='all')


df.Sex.value_counts()
df.Sex.value_counts(normalize = True)
df[df.Survived != -888].Survived.value_counts()
df.Pclass.value_counts()
#Visualize Sex count,Survived and Class wise survival

df.Sex.value_counts().plot(kind='bar');
df[df.Survived != -888].Survived.value_counts().plot(kind='bar');
df.Pclass.value_counts().plot(kind='bar');




df.Pclass.value_counts().plot(kind='bar',rot = 0,title = "Pclass count on Titanic");
#for univariate distributions we use Histogram and KDE

#KDE stands for Kerenl Density Estimation
df.Age.plot(kind ='hist',title = 'histogram for Age' );
df.Age.plot(kind ='kde',title = 'histogram for Age' );
df.Fare.plot(kind ='hist',title = 'histogram for Age' );
#We use bivariate distribution for Scatter plot
df.plot.scatter(x='Age',y='Fare',title='Scatter Plot:Age vs Fare');
df.plot.scatter(x='Age',y='Fare',title='Scatter Plot:Age vs Fare',alpha = 0.5);
df.plot.scatter(x='Pclass',y='Fare',title='Scatter Plot:Pclass vs Fare');
df.groupby('Sex').Age.median()
#group by

df.groupby('Pclass').Fare.median()






df.groupby('Pclass').Age.median()
df.groupby(['Pclass'])['Fare','Age'].median()






df.groupby(['Pclass']).agg({'Fare':'mean','Age':'median'})
# more complicated aggregation

aggregations ={

    'Fare': {#work on fare column

        'mean_Fare':'mean',

        'median_Fare':'median',

        'Max_Fare':max,

        'Min_Fare' :np.min

    },

    'Age': {

        'mean_Age':'mean',

        'median_Age':'median',

        'Max_Age':max,

        'Min_Age' :np.min

    }

}
df.groupby(['Pclass']).agg(aggregations)
df.groupby(['Pclass','Embarked']).Fare.median()
pd.crosstab(df.Sex,df.Pclass)
pd.crosstab(df.Sex,df.Pclass).plot(kind='bar');
#pivot table

df.pivot_table(index='Sex',columns = 'Pclass',values = 'Age',aggfunc='mean')
df.groupby(['Sex','Pclass']).Age.mean().unstack()
df.isnull().sum()
train_df.isnull().sum()
df.info()
df[df.Embarked.isnull()]
#how many people embarked at a particular points

df.Embarked.value_counts()
#which embarked point has highest survival count

pd.crosstab(df[df.Survived != -888].Survived,df[df.Survived != -888].Embarked)
# impute missing value with 'S'

#df.loc[df.Embarked.isnull(),'Embarked'] = S

#df.Embarked.fillna('S',inplace = True)
df.groupby(['Pclass','Embarked']).Fare.median() 


df.Embarked.fillna('C',inplace = True)
df.Embarked.isnull().sum()
df.info()
df[df.Fare.isnull()]
median_fare = df.loc[(df.Pclass == 3) & (df.Embarked=='S'),'Fare'].median()

print (median_fare)
df.Fare.fillna(median_fare,inplace=True)
df.info()
df.Age.isnull().sum()
df.Age.plot(kind='hist',bins=20);
df.Age.mean()
df.groupby('Sex').Age.median()
df[df.Age.notnull()].boxplot('Age','Sex');
df[df.Age.notnull()].boxplot('Age','Pclass');
df.Name.head()
def GetTitle(name):

    first_name_with_title = name.split(',')[1]

    title = first_name_with_title.split('.')[0]

    title = title.strip().lower()

    return title
#use map function to apply the function on each Name value row i

df.Name.map(lambda x : GetTitle(x))
df.Name.map(lambda x : GetTitle(x)).unique()
def GetTitle(name):

    title_group = {'mr' : 'Mr', 

               'mrs' : 'Mrs', 

               'miss' : 'Miss', 

               'master' : 'Master',

               'don' : 'Sir',

               'rev' : 'Sir',

               'dr' : 'Officer',

               'mme' : 'Mrs',

               'ms' : 'Mrs',

               'major' : 'Officer',

               'lady' : 'Lady',

               'sir' : 'Sir',

               'mlle' : 'Miss',

               'col' : 'Officer',

               'capt' : 'Officer',

               'the countess' : 'Lady',

               'jonkheer' : 'Sir',

               'dona' : 'Lady'

                 }

    first_name_with_title = name.split(',')[1]

    title = first_name_with_title.split('.')[0]

    title = title.strip().lower()

    return title_group[title]
df['Title'] = df.Name.map(lambda x : GetTitle(x))
df.head()
df[df.Age.notnull()].boxplot('Age','Title');
#replace missing values

title_age_median = df.groupby('Title').Age.transform('median')
df.Age.fillna(title_age_median,inplace = True)
df.info()
# use histograms to get to understand the distribution

df.Age.plot(kind = 'hist' , bins = 20);
df.loc[df.Age > 70]
# hsitograms for fare

df.Fare.plot(kind='hist',bins = 20, title = 'Histograms for Fare')
df.Fare.plot(kind='box');
# look for the outliers

df.loc[df.Fare == df.Fare.max()]
#try to use transformation to reduce the skewness
LogFare = np.log(df.Fare +1)#adding 1 to accomalate 
LogFare.plot(kind='hist',bins = 20);
#binning
pd.qcut(df.Fare,4)
pd.qcut(df.Fare,4,labels=['very_low','low','high','very_high'])
pd.qcut(df.Fare,4,labels = ['very_low','low','high','very_high']).value_counts().plot(kind='bar',rot = 0);
# create fare bin feature

df['Fare_Bin']=pd.qcut(df.Fare,4,labels=['very_low','low','high','very_high'])
#Age State based on Age

df['AgeState'] = np.where(df['Age']>=18,'Adult','Child')
#AgeState Counts

df['AgeState'].value_counts()
pd.crosstab(df[df.Survived != -888].Survived , df[df.Survived != -888].AgeState)
df['FamilySize'] = df.Parch + df.SibSp + 1 # i for Self
#explore the family feature

df['FamilySize'].plot(kind = 'hist',color = 'c');
#further exploring familoy size with mjax family size

df.loc[df.FamilySize == df.FamilySize.max(),['Name','Survived','FamilySize','Ticket']]
pd.crosstab(df[df.Survived != -888].Survived , df[df.Survived != -888].FamilySize)
# a lady aged 18 or more who has Parch >0 and is married 

df['IsMother'] = np.where(((df.Sex == 'female') & (df.Parch > 0) & (df.Age>18) & (df.Title != 'Miss')),1,0)




#Crosstab with mother

pd.crosstab(df[df['Survived'] != -888].Survived,df[df['Survived'] != -888].IsMother )
#explore Cabin values

df.Cabin
#Getting unique cabin

df.Cabin.value_counts()
#We see that T is odd one out in above observation so we can asume it is mistaken value
# get the value to Nan

df.loc[df.Cabin == 'T','Cabin']=np.NaN
def get_deck(cabin):

    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')

df['Deck'] = df['Cabin'].map(lambda x: get_deck(x))
# check counts

df.Deck.value_counts()
pd.crosstab(df[df.Survived != -888].Survived,df[df.Survived != -888].Deck)
df.info()
#sex

df['IsMale'] = np.where(df.Sex == 'male',1,0)
#columns deck,pclass,title,Agestate

df = pd.get_dummies(df,columns=['Deck','Pclass','Title','Fare_Bin','Embarked','AgeState'])
df.info()






#drop columns
df.drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'],axis = 1,inplace = True)
#reorder columns

columns = [column for column in df.columns if column != 'Survived']

columns = ['Survived']+columns

df = df[columns]
df.info()
df.to_csv('out.csv')#Saving Dataset before making predicting model

#This would be saved as output in Version folder.
train_df = df.loc[0:891,:]
train_df.info()
train_df.tail()
test_df = df.loc[892:,:]
test_df.tail()
train_df.shape
test_df.shape

test_df.info()

X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("Survived", axis=1).copy()
X_train.shape
Y_train.shape
X_test.shape
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
Xtr = X_train.copy()

Xtr.head()
ytr = Y_train.copy()

#target column i.e price range

ytr.head()

Y_train.head()
#apply SelectKBest class to extract top 20 best features

bestfeatures = SelectKBest(score_func=chi2, k=20)

fit = bestfeatures.fit(Xtr,ytr)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(Xtr.columns)
#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(32,'Score'))  #print 10 best features
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt
model = ExtraTreesClassifier()

model.fit(Xtr,ytr)






print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=Xtr.columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.show()





train_df.head()
corrmat = train_df.corr()

print(corrmat.Survived)

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(train_df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
list = ['Title_Master','Fare_Bin_low','Title_Lady','Fare_Bin_high','Deck_F','Title_Sir','AgeState_Adult','Deck_A','FamilySize','Deck_G',

        'Title_Officer','Embarked_Q','Deck_B','IsMother','Embarked_C','Deck_Z','Deck_D','Deck_E','AgeState_Child']
X_train.drop(list,axis=1,inplace = True)

X_test.drop(list,axis = 1,inplace = True)

X_train.shape

Y_train.shape
# Random FOrest Classifier using Grid SearchSearch CV
X_test.shape
rfc=RandomForestClassifier(random_state=42)
param_grid = { 

    'n_estimators': [200, 500],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8],

    'criterion' :['gini', 'entropy']

}
"""CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

CV_rfc.fit(X_train, Y_train)"""
"""CV_rfc.best_params_"""
rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 500, max_depth=7, criterion='gini')
rfc1.fit(X_train, Y_train)
pred=rfc1.predict(X_test)
X_test.index
df_result = pd.DataFrame(pred)
df_result
df_result['Survived'] = pred
df_result.drop(0,axis =1,inplace = True)
df_result['PassengerId']=X_test.index
df_result.head()

df_result = df_result.set_index('PassengerId')
df_result.head()
df_result.to_csv('output.csv')

from sklearn.model_selection import RandomizedSearchCV



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10,20]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4,10,15,20,30]

# Method of selecting samples for training each tree

bootstrap = [True, False]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
"""

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42,

                               n_jobs = -1)



# Fit the random search model

rf_random.fit(X_train, Y_train)"""
"""print(rf_random.best_params_)"""
rf1=RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators= 200, max_depth=7, criterion='gini',

                           min_samples_split = 2, min_samples_leaf = 2, bootstrap =  False)
rf1.fit(X_train, Y_train)
pred1 = rf1.predict(X_test)
df_result_1 = pd.DataFrame(pred1)
df_result_1
df_result_1['Survived'] = pred1
df_result_1.head()
df_result_1.drop(0,axis =1,inplace = True)
df_result_1['PassengerId']=X_test.index
df_result_1.head()
df_result_1 = df_result_1.set_index('PassengerId')
df_result_1.head()
df_result_1.to_csv('output_1.csv')
