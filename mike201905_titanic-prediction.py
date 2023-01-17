# Predict survival on the Titanic

# Import libararies

# linear algebra

import numpy as np

# data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



# Data science packages

from scipy import stats

from scipy.stats import norm

# Data Preprocessing

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

# Visualization Libraries

import matplotlib.pyplot as plt

import seaborn as sns

# Import Models

# Binary classification model(https://stackabuse.com/classification-in-python-with-scikit-learn-and-pandas/)

# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model

from sklearn.linear_model import LogisticRegression

# https://scikit-learn.org/stable/modules/ensemble.html

from sklearn.ensemble import RandomForestClassifier

# https://qiita.com/kazuki_hayakawa/items/18b7017da9a6f73eba77

# https://qiita.com/arata-honda/items/8d08f31aa7d7cbae4c91

from sklearn import svm

# https://spjai.com/neural-network-parameter/

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

from sklearn.neural_network import MLPClassifier



# Base class for all estimators in scikit-learn

from sklearn.base import BaseEstimator

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier

# Mixin class for all classifiers in scikit-learn

from sklearn.base import ClassifierMixin

# Accuracy = (TP + TN) / (TP + TN + FP + FN)

from sklearn.metrics import accuracy_score

# Cross-validation

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_predict



# utilities

from datetime import datetime

# Disable warning output

import warnings

warnings.filterwarnings('ignore')

# Show matplot graph

%matplotlib inline
# Read data from train dataset

df_train = pd.read_csv('../input/train.csv')

#check the columns

print(df_train.columns)

print(df_train.dtypes)

# Read data from test dataset

df_test = pd.read_csv('../input/test.csv')

print(df_test.columns)
df_train.info()
df_train.describe()
fig = plt.figure(figsize=(18, 12))

fig.set(alpha=0.2)  # alpha for graph color



plt.subplot2grid((2,3),(0,0))             # subplot in one parent plot

df_train.Survived.value_counts().plot(kind='bar')# bar graph 

plt.title(u"Survive (1:Survived)") # title

plt.ylabel(u"Passenger Count")  



plt.subplot2grid((2,3),(0,1))

df_train.Pclass.value_counts().plot(kind="bar")

plt.ylabel(u"Passenger Count")

plt.title(u"Pclass")



plt.subplot2grid((2,3),(0,2))

plt.scatter(df_train.Survived, df_train.Age)

plt.ylabel(u"Age")                         # y label name

plt.grid(b=True, which='major', axis='y') 

plt.title(u"Survive (1:Survived) by Age")





plt.subplot2grid((2,3),(1,0), colspan=2)

df_train.Age[df_train.Pclass == 1].plot(kind='kde')   

df_train.Age[df_train.Pclass == 2].plot(kind='kde')

df_train.Age[df_train.Pclass == 3].plot(kind='kde')

plt.xlabel(u"Age")# plots an axis lable

plt.ylabel(u"Density") 

plt.title(u"Age distribution by pclass")

plt.legend((u'p1', u'p2',u'p3'),loc='best') # sets our legend for our graph.





plt.subplot2grid((2,3),(1,2))

df_train.Embarked.value_counts().plot(kind='bar')

plt.title(u"Passenger Count by Embarked")

plt.ylabel(u"Passenger Count")

sns.set()

plt.show()
#Passenger Suvived by Pclass

fig = plt.figure(figsize=(8, 6))

fig.set(alpha=0.2)  # alpha for graph color



Survived_0 = df_train.Pclass[df_train.Survived == 0].value_counts()

Survived_1 = df_train.Pclass[df_train.Survived == 1].value_counts()

df=pd.DataFrame({u'Survived':Survived_1, u'Not Survived':Survived_0})

df.plot(kind='bar', stacked=True)

plt.title(u"Passenger Suvived by Pclass")

plt.xlabel(u"Pclass") 

plt.ylabel(u"Passenger Count") 

plt.show()
#Passenger Suvived by Sex

fig = plt.figure(figsize=(8, 6))

fig.set(alpha=0.2)  # alpha for graph color



Survived_m = df_train.Survived[df_train.Sex == 'male'].value_counts()

Survived_f = df_train.Survived[df_train.Sex == 'female'].value_counts()

df=pd.DataFrame({u'Male':Survived_m, u'Female':Survived_f})

df.plot(kind='bar', stacked=True)

plt.title(u"Passenger Survived by Sex")

plt.xlabel(u"Sex") 

plt.ylabel(u"Passenger Count")

plt.show()
#Passenger Suvived by combined Pclass and Sex

fig=plt.figure(figsize=(18, 16))

fig.set(alpha=0.65) # alpha for graph color

plt.title(u"Passenger Suvived by combined Pclass and Sex")



ax1=fig.add_subplot(141)

df_train.Survived[df_train.Sex == 'female'][df_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')

ax1.set_xticklabels([u"Survived", u"Not Survived"], rotation=0)

ax1.legend([u"Female/High Class"], loc='best')



ax2=fig.add_subplot(142, sharey=ax1)

df_train.Survived[df_train.Sex == 'female'][df_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')

ax2.set_xticklabels([u"Not Survived", u"Survived"], rotation=0)

plt.legend([u"Female/Low Class"], loc='best')



ax3=fig.add_subplot(143, sharey=ax1)

df_train.Survived[df_train.Sex == 'male'][df_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')

ax3.set_xticklabels([u"Not Survived", u"Survived"], rotation=0)

plt.legend([u"Male/High Class"], loc='best')



ax4=fig.add_subplot(144, sharey=ax1)

df_train.Survived[df_train.Sex == 'male'][df_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')

ax4.set_xticklabels([u"Not Survived", u"Survived"], rotation=0)

plt.legend([u"Male/Low Class"], loc='best')

sns.set()

plt.show()
#Passenger Suvived by Embarked

fig = plt.figure(figsize=(10, 8))

fig.set(alpha=0.2)  # alpha for graph color



Survived_0 = df_train.Embarked[df_train.Survived == 0].value_counts()

Survived_1 = df_train.Embarked[df_train.Survived == 1].value_counts()

df=pd.DataFrame({u'Survived':Survived_1, u'Not Survived':Survived_0})

df.plot(kind='bar', stacked=True)

plt.title(u"Passenger Suvived by Embarked")

plt.xlabel(u"Embarked") 

plt.ylabel(u"Passenger Count") 



plt.show()
#Passenger Suvived by cabin

fig = plt.figure(figsize=(10, 8))

fig.set(alpha=0.2)  # alpha for graph color



Survived_cabin = df_train.Survived[pd.notnull(df_train.Cabin)].value_counts()

Survived_nocabin = df_train.Survived[pd.isnull(df_train.Cabin)].value_counts()

df=pd.DataFrame({u'Yes':Survived_cabin, u'No':Survived_nocabin}).transpose()

df.plot(kind='bar', stacked=True)

plt.title(u"Survived by cabin")

plt.xlabel(u"If have cabin") 

plt.ylabel(u"Passenger Count")

plt.show()
#https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

#https://blog.csdn.net/m0_38024592/article/details/80836217

#combine train dataset and test dataset

df_all = pd.concat([df_train,df_test],axis=0).reset_index(drop=True)
# 0, Pre-Process for Name

# Create feature Title based on variable Name 

df_all['Title'] = df_all['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])



# Group low-occuring, related titles together

df_all['Title'][df_all.Title == 'Jonkheer'] = 'Master'

df_all['Title'][df_all.Title.isin(['Ms','Mlle'])] = 'Miss'

df_all['Title'][df_all.Title == 'Mme'] = 'Mrs'

df_all['Title'][df_all.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'

df_all['Title'][df_all.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
print(df_all['Title'].value_counts())
# # Pre-Process for Cabin

# # split one row to muliple rows for carbin because some records have multiple valus in Cabin column 

# df_all=df_all.drop(['Cabin'], axis=1).join(df_all['Cabin'].str.split(' ', expand=True).stack().reset_index(level=1, drop=True).rename('Cabin'))

# # Create binary features for each deck

# df_all['Deck'] = df_all['Cabin'].str[0]

# # Create feature for the room number

# df_all['Room'] = df_all['Cabin'].str[1:]

# print("Column Deck\n", df_all['Deck'])

# print("Column Room\n", df_all['Room'])
###########################

# Pre-Process for Ticket

# split Ticket column to Ticket class and Ticket Num 

############################

# http://www.datasciencemadesimple.com/string-replace-column-dataframe-python/

# split Ticket to multiple column

df_all = pd.concat([df_all, df_all['Ticket'].str.split(' ', expand=True)], axis=1).drop('Ticket', axis=1)

df_all.rename(columns={0: 'Ticket_Cls', 1: 'Ticket_Num'}, inplace=True)

# set Ticket Number without Ticket class to variable Ticket_Num

df_all['Ticket_Num'][df_all['Ticket_Num'].isnull()]=df_all['Ticket_Cls']

df_all['Ticket_Num']=pd.to_numeric(df_all['Ticket_Num'], errors='coerce')

# set Ticket class to None if the value is numeric in Ticket_Cls variable

#print(df_all['Ticket_Cls'][map(lambda x: x.isdigit(), df_all['Ticket_Cls'])])

df_all['Ticket_Cls'][df_all['Ticket_Cls'].str.isnumeric()]='None'

print("Ticket_Cls\n",df_all['Ticket_Cls'])

print("Ticket_Num\n",df_all['Ticket_Num'])
#1, Check Nan data

#Missing data for train dataset

#two Important things for missing data:

#a. How prevalent is the missing data?

#b.Is missing data random or does it have a pattern?

#https://note.nkmk.me/python-pandas-nan-judge-count/

df_all.reset_index(drop=True,inplace=True)

total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])

missing_data
#2, Process Nan values

#2

#Remove feature 2 because too many nan

df_all = df_all.drop(2, axis=1)



#Cabin

#Remove feature Cabin because feature Deck and room have been created

df_all = df_all.drop(['Cabin'], axis=1)



# #Deck

# #Replace to None

# df_all['Deck'].fillna('None',inplace=True) 



# #Room

# #Replace to mean(better use mean for numerical variable)

# df_all['Room']=pd.to_numeric(df_all['Room'], errors='coerce')

# df_all['Room'].fillna(df_all['Room'].mean(),inplace=True) 



#Embarked

#Replace to most common value (better use most frequently values for categorical variable )

df_all['Embarked'].fillna(df_all['Embarked'].dropna().mode().values.item(),inplace=True) 



#Fare

#Replace to mean(better use mean for numerical variable)

df_all['Fare'].fillna(df_all['Fare'].median(),inplace=True) 



#Fare

#Replace to mean(better use mean for numerical variable)

df_all['Ticket_Num'].fillna(df_all['Ticket_Num'].median(),inplace=True) 
#Replace Nan to (mean -std) or (mean + std) by random for Age

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))

axis1.set_title('Original Age values')

df_all['Age'][df_all.Age.isnull()==False].plot(kind='hist', bins=70, ax=axis1)



#print(pd.concat([df_all['Age'],df_all['Title']],axis=1).groupby('Title').agg(['count','mean']))

age_median_per_title=pd.concat([df_all['Age'],df_all['Title']],axis=1).groupby('Title').median().reset_index()

for index,row in age_median_per_title.iterrows():

    age_title=row['Title']

    age_median =row['Age']

    #It seems inplace=True is not working if there hase filter condition

    df_all['Age'][df_all['Title']==age_title]=df_all['Age'][df_all['Title']==age_title].fillna(age_median)

    

axis2.set_title('New Age values')

df_all['Age'][df_all.Age.isnull()==False].plot(kind='hist', bins=70, ax=axis2)

plt.show()

df_all.isnull().sum().max() #check if missing data exists
# Create a new variable Age_Cls based on Age

df_all['Age_Cls']=0

df_all['Age_Cls'][df_all['Age']<=10] = 0

df_all['Age_Cls'][(df_all['Age']>10) & (df_all['Age']<=20)] = 1

df_all['Age_Cls'][(df_all['Age']>20) & (df_all['Age']<=30)] = 2

df_all['Age_Cls'][(df_all['Age']>30) & (df_all['Age']<=40)] = 3

df_all['Age_Cls'][(df_all['Age']>40) & (df_all['Age']<=50)] = 4

df_all['Age_Cls'][(df_all['Age']>50) & (df_all['Age']<=60)] = 5

df_all['Age_Cls'][df_all['Age']>60] = 6
# qcut() creates a new variable that identifies the quartile range of Fare

df_all['Fare_Bin'] = pd.qcut(df_all['Fare'], 6, labels=['Fare_G1', 'Fare_G2', 'Fare_G3', 'Fare_G4', 'Fare_G5', 'Fare_G6'])
df_all.info()
#4, Check variables which only have Unique values or discrete values for train data

#https://note.nkmk.me/python-pandas-value-counts/

print(df_all.nunique()/len(df_all))

#Process variables which only have Unique values

#PassengerId only have unique value

#Remove PassengerId from all of the dataset

#df_all.drop(['PassengerId'],axis=1,inplace=True)

#Name only have unique

#Remove Name from all of the dataset

df_all.drop(['Name'],axis=1,inplace=True)
# 2-2-1, Scaling

#scale numerical variable room

#For Age,Fare,Room,Ticket_Num

#remapping the values to small range [-1,1] or [0,1]

#StandardScaler will subtract the mean from each value then scale to the unit variance

scaler = preprocessing.StandardScaler()

df_all['Fare'] = scaler.fit_transform(df_all['Fare'].values.reshape(-1, 1))

df_all['Age'] = scaler.fit_transform(df_all['Age'].values.reshape(-1, 1))

# df_all['Room'] = scaler.fit_transform(df_all['Room'].values.reshape(-1, 1))

df_all['Ticket_Num'] = scaler.fit_transform(df_all['Ticket_Num'].values.reshape(-1, 1))
#6, Check Data Quality

#1),Normality

#2),Linearity



#Age

#histogram and normal probability plot

sns.distplot(df_all[df_all['Age']>0]['Age'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_all[df_all['Age']>0]['Age'], plot=plt)

# It seems not bad, keep no change
#Fare

#histogram and normal probability plot

sns.distplot(df_train['Fare'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['Fare'], plot=plt)

#It's very peakedness and do log transformation 
df_all.info()
#7, dummy variables

#convert categorical variable into dummy

#https://note.nkmk.me/python-pandas-get-dummies/

#Ticket Variable has lots of different valus, not sure if should use

#In additional convert some numerical variable which have categorical attribute into dummy

#Drop ticket_num

df_all.drop(['Ticket_Num'],axis=1,inplace=True)

#Drop ticket_Cls

df_all.drop(['Ticket_Cls'],axis=1,inplace=True)

#Drop Fare

#df_all.drop(['Fare'],axis=1,inplace=True)

#Drop Age

#df_all.drop(['Age'],axis=1,inplace=True)



#Change dtype to object

df_all['Pclass'] = df_all['Pclass'].astype(object)

df_all['SibSp'] = df_all['SibSp'].astype(object)

df_all['Parch'] = df_all['Parch'].astype(object)

# df_all['Deck'] = df_all['Deck'].astype(object)

df_all['Age_Cls'] = df_all['Age_Cls'].astype(object)

df_all = pd.get_dummies(df_all)

print(df_all.columns)
df_all.info()
print('Create DataSet', datetime.now(), )

#Create Train dataset, Label, Test dataset

#http://ailaby.com/lox_iloc_ix/

#https://note.nkmk.me/python-pandas-at-iat-loc-iloc/

df_all.reset_index(drop=True,inplace=True)

y = df_all['Survived'][df_all['Survived'].isnull() == False]

x_train = df_all[:len(y)].drop(['PassengerId','Survived'],axis=1).values

x_test = df_all[len(y):].drop(['PassengerId','Survived'],axis=1).values



x_train.shape, x_test.shape, y.shape
#Algorithm (fx()= w*x +b , 1 if >=0 else 0)

#Perceptron is a single layer neural network and a multi-layer perceptron is called Neural Networks.

#Perceptron is a linear classifier (binary), it is used in supervised learning.

#It helps to classify the given input data.

class PerceptronClassification(BaseEstimator,ClassifierMixin):

    """ Perceptron Classifier

    Parameters

    ------------

    rate : float

        Learning rate (ranging from 0.0 to 1.0)

    number_of_iteration : int

    Number of iterations over the input dataset.



    Attributes:

    ------------

    weight_matrix : 1d-array

        Weights after fitting.



    error_matrix : list

        Number of misclassification in every epoch(one full training cycle on the training set)

    """

    def __init__(self, rate = 0.01, number_of_iterations = 100):

        self.rate = rate

        self.number_of_iterations = number_of_iterations



    def fit(self, X, y):

        """ Fit training data

        Parameters:

        ------------

        X : array-like, shape = [number_of_samples, number_of_features]

            Training vectors.

        y : array-like, shape = [number_of_samples]

            Target values.

        Returns

        ------------

        self : object

        """

        self.weight_matrix = np.zeros(1 + X.shape[1])

        self.errors_list = []



        for _ in range(self.number_of_iterations):

            errors = 0

            for xi, target in zip(X, y):

                update = self.rate * (target - self.predict(xi))

                self.weight_matrix[1:] += update * xi

                self.weight_matrix[0] += update

                errors += int(update != 0.0)

            self.errors_list.append(errors)

        return self



    def dot_product(self, X):

        """ Calculate the dot product """

        return (np.dot(X, self.weight_matrix[1:]) + self.weight_matrix[0])



    def predict(self, X):

        """ Predicting the label for the input data """

        return np.where(self.dot_product(X) >= 0.0, 1, 0)
# Cross validation with 100 iterations to get smoother mean test and train

# score curves, each time with 25% data randomly selected as a validation set.

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=10)

stratifiedkfold = StratifiedKFold(n_splits=10)
print('Define Classification model', datetime.now(), )

#Prediction by Perceptron Algorithm

perceptron_classifiter=PerceptronClassification(number_of_iterations = 3000)
#Logistic Regression

#Logistic Regression is a type of Generalized Linear Model (GLM) 

#that uses a logistic function to model a binary variable based on any kind of independent variables.

LR = LogisticRegression(random_state=10, solver='lbfgs', multi_class='ovr')
ADA = AdaBoostClassifier()
#Support Vector Machines (SVMs)

#a type of classification algorithm that are more flexible

#they can do linear classification, but can use other non-linear basis functions.

SVM = svm.LinearSVC()  
SVM.get_params().keys()
# define parameters as dict type

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

grid_search_SVM = GridSearchCV(SVM, param_grid, cv=5)
#Random Forests are an ensemble learning method

#that fit multiple Decision Trees on subsets of the data and average the results.

#0.856502 (2000)

#0.78947 (learnboard)

#rf_parameters = {'n_estimators': 2000, 'min_samples_split': 5, 'random_state':5}

rf_parameters = {'n_estimators': 2000, 'max_depth': 30, 'random_state':10}

# RF = RandomForestClassifier(n_estimators=3000, max_depth=20, random_state=5)

RF = RandomForestClassifier(**rf_parameters)  
#Multi-layer Perceptron classifier.

#This model optimizes the log-loss function using LBFGS or stochastic gradient descent.

#neural_network.MLPClassifier¶

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000, 2), random_state=100)  
# GradientBoostingClassifier

GBR = GradientBoostingClassifier(n_estimators=2000, max_depth=30, max_leaf_nodes=8, min_samples_split=5, random_state =10)
# define parameters as dict type

param_grid = {'n_estimators': [2000], 'max_depth' : [20], 'random_state' : [0, 5,10]}

grid_search = GridSearchCV(RF, param_grid, cv=5)
bagging = BaggingClassifier(n_estimators=2000, max_samples=1.0, max_features=1.0, n_jobs=-1, random_state =10)

# bagging_LR.fit(x_train, y_train)

# print(bagging_LR.score(x_val,y_val))

# predictions = bagging_LR.predict(x_test)
# Build a Extra forest

Extraforest = ExtraTreesClassifier(n_estimators=2000, random_state=10)
# votemodel = VotingClassifier(estimators=[('LR', LR), ('RF', RF), ('GBR', GBR)],voting='soft'])

votemodel = VotingClassifier(estimators=[('LR', LR), ('RF', RF), ('GBR', GBR), ('ADA', ADA), ('Extraforest', Extraforest), 

                                         ('bagging_Decision tree', bagging)],voting='hard',weights=[1,1,1,1,1,1])

#votemodel = VotingClassifier(estimators=[('LR', LR), ('RF', RF), ('GBR', GBR), ('ADA', ADA)],voting='hard',weights=[1,1,1,1])
print('Fit and TEST score for each model',datetime.now(), )

#names = ["Logistic Regression", "Support Vector Machines", "Random Forests", "Multi-layer Perceptron classifier","Gradient Boosting",'Voting Classifier']

names = ["Logistic Regression", "Random Forests", "Gradient Boosting", "ADA", "Extraforest", "bagging", "Voting Classifier"]

models= [LR, RF, GBR, ADA, Extraforest, bagging, votemodel]

#names = ["Random Forests"]

#models= [RF]

for name, model in zip(names, models):

    model.fit(x_train, y)

    print('model {} start'.format(name))

    scores = cross_val_score(model, x_train, y,cv=stratifiedkfold)

    # 各分割におけるスコア

    print('{} Cross-Validation scores: {}'.format(name,scores))

    # スコアの平均値

    score_mean = np.mean(scores)

    print('{} Cross-Validation Average score: {:.6f}'.format(name,score_mean))

    #print(cross_val_predict(model, x_train, y,cv=stratifiedkfold))
# print('Predict submission', datetime.now(),)

# submission_PER=pd.DataFrame({'PassengerId':df_test['PassengerId'], 'Survived':perceptron_classifiter.predict(x_test)})

# submission_LR=pd.DataFrame({'PassengerId':df_test['PassengerId'], 'Survived':LR.predict(x_test)})

# submission_NN=pd.DataFrame({'PassengerId':df_test['PassengerId'], 'Survived':NN.predict(x_test)})

# submission_SVM=pd.DataFrame({'PassengerId':df_test['PassengerId'], 'Survived':SVM.predict(x_test)})

#submission_RF=pd.DataFrame({'PassengerId':df_all['PassengerId'][df_all['Survived'].isnull()], 'Survived':RF.predict(x_test).astype(int)})

submission_bagging=pd.DataFrame({'PassengerId':df_all['PassengerId'][df_all['Survived'].isnull()], 'Survived':bagging.predict(x_test).astype(int)})

submission_votemodel=pd.DataFrame({'PassengerId':df_all['PassengerId'][df_all['Survived'].isnull()], 'Survived':votemodel.predict(x_test).astype(int)})
# submission_RF = submission_RF.drop_duplicates(subset='PassengerId', keep='first')

# submission_RF = submission_RF.reset_index(drop=True)

#submission_GBR = submission_GBR.drop_duplicates(subset='PassengerId', keep='first')

submission_bagging = submission_bagging.reset_index(drop=True)

submission_votemodel = submission_votemodel.reset_index(drop=True)
# #output to submission csv

# #0.72727

# submission_PER.to_csv("submission_v2_PER.csv", index=False)

# #0.77511

# submission_LR.to_csv("submission_v2_LR.csv", index=False)

# #0.76555

# submission_NN.to_csv("submission_v2_NN.csv", index=False)

# #0.76076

# submission_SVM.to_csv("submission_v2_SVM.csv", index=False)

# #0.77511

#submission_RF.to_csv("submission_v3_RF.csv", index=False)

submission_votemodel.to_csv("submission_v4_vm.csv", index=False)

#submission_RF.to_csv("submission_v4_RF.csv", index=False)

submission_bagging.to_csv("submission_v4_bagging.csv", index=False)

print('Save submission', datetime.now(),)
# Learning curve

# Visualization model if overfitting or underfitting

# not good for generalization ability if overfitting

# https://blog.csdn.net/limiyudianzi/article/details/79626702

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    """

    Generate a simple plot of the test and training learning curve.



    Parameters

    ----------

    estimator : object type that implements the "fit" and "predict" methods

        An object of that type which is cloned for each validation.



    title : string

        Title for the chart.



    X : array-like, shape (n_samples, n_features)

        Training vector, where n_samples is the number of samples and

        n_features is the number of features.



    y : array-like, shape (n_samples) or (n_samples, n_features), optional

        Target relative to X for classification or regression;

        None for unsupervised learning.



    ylim : tuple, shape (ymin, ymax), optional

        Defines minimum and maximum yvalues plotted.



    cv : int, cross-validation generator or an iterable, optional

        Determines the cross-validation splitting strategy.

        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,

          - integer, to specify the number of folds.

          - :term:`CV splitter`,

          - An iterable yielding (train, test) splits as arrays of indices.



        For integer/None inputs, if ``y`` is binary or multiclass,

        :class:`StratifiedKFold` used. If the estimator is not a classifier

        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.



        Refer :ref:`User Guide <cross_validation>` for the various

        cross-validators that can be used here.



    n_jobs : int or None, optional (default=None)

        Number of jobs to run in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.

        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`

        for more details.



    train_sizes : array-like, shape (n_ticks,), dtype float or int

        Relative or absolute numbers of training examples that will be used to

        generate the learning curve. If the dtype is float, it is regarded as a

        fraction of the maximum size of the training set (that is determined

        by the selected validation method), i.e. it has to be within (0, 1].

        Otherwise it is interpreted as absolute sizes of the training sets.

        Note that for classification the number of samples usually have to

        be big enough to contain at least one sample from each class.

        (default: np.linspace(0.1, 1.0, 5))

    """

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    print("Training size:{}".format(train_sizes))

    print("Train score: {}".format(train_scores_mean))

    print("Cross-validation score: {}".format(test_scores_mean))

    return plt
X = x_train

y = y



# RandomForest

#rf_parameters = {'n_jobs': -1, 'n_estimators': 2000, 'warm_start': True, 'max_depth': 20, 'min_samples_leaf': 2, 'max_features' : 'sqrt','verbose': 0}

#rf_parameters = {'n_estimators': 1000, 'max_depth': 30}



# AdaBoost

ada_parameters = {'n_estimators':500, 'learning_rate':0.1}



# ExtraTrees

et_parameters = {'n_jobs': -1, 'n_estimators':500, 'max_depth': 8, 'min_samples_leaf': 2, 'verbose': 0}



# GradientBoosting

gb_parameters = {'n_estimators': 500, 'max_depth': 5, 'min_samples_leaf': 2, 'verbose': 0}



# DecisionTree

dt_parameters = {'max_depth':8}



# KNeighbors

knn_parameters = {'n_neighbors':2}



# SVM

svm_parameters = {'kernel':'linear', 'C':0.025}



# XGB

gbm_parameters = {'n_estimators': 2000, 'max_depth': 4, 'min_child_weight': 2, 'gamma':0.9, 'subsample':0.8, 

                  'colsample_bytree':0.8, 'objective': 'binary:logistic', 'nthread':-1, 'scale_pos_weight':1}

title = "Learning Curves (RandomForest)"

plot_learning_curve(votemodel, title, X, y, ylim=(0.5, 1.01), cv=stratifiedkfold,  n_jobs=4, train_sizes=[200, 300, 400, 500, 600])

# plot_learning_curve(RandomForestClassifier(**rf_parameters), title, X, y, ylim=(0.5, 1.01), cv=cv,  n_jobs=4, train_sizes=[200, 250, 350, 400, 450, 500,550,600,650])

# plot_learning_curve(votemodel, title, X, y, ylim=(0.5, 1.01), cv=cv,  n_jobs=4, train_sizes=[200, 250, 350, 400, 450, 500,550,600,650])

plt.show()
##########################

# Check importance of features

##########################

features_list = df_all.drop(['Survived','PassengerId'],axis=1).columns.values



# Fit a random forest with (mostly) default parameters to determine feature importance

forest = RandomForestClassifier(**rf_parameters)

forest.fit(X, y)

feature_importance = forest.feature_importances_



# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())



print(feature_importance)

# Get the indexes of all features over the importance threshold

important_idx = np.where(feature_importance)[0]



# Get the sorted indexes of important features

sorted_idx = np.argsort(feature_importance[important_idx])[::-1]

print("\nFeatures sorted by importance (DESC):\n", features_list[sorted_idx])



# Adapted from http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html

pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(18, 18))

plt.subplot(1, 1, 1)

plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], align='center')

plt.yticks(pos, features_list[sorted_idx[::-1]])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()