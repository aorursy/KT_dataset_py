# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



print ('sum of train data', df_train.isnull().sum())

print ('')

print ('sum of test data', df_test.isnull().sum())
###################################################################

# delete feature,who misses too many data

# delete feature, who doesn't contain the meaningful information

###################################################################

#1 - train data

# too many data miss for feature Cabin, 

df_train = df_train.drop('Cabin', 1)

# feature - Ticket, Name, doesn't contain the meaningful information for the algorithm

df_train = df_train.drop('Name', 1)

df_train = df_train.drop('Ticket', 1)

#df_train = df_train.drop('PassengerId',1)





#2 - test data

# too many data miss for feature Cabin, 

df_test = df_test.drop('Cabin', 1)

# feature - Ticket, Name, doesn't contain the meaningful information for the algorithm

df_test = df_test.drop('Name', 1)

df_test = df_test.drop('Ticket', 1)

#df_test = df_test.drop('PassengerId', 1)



print ('sum of train data', df_train.isnull().sum())

print ('')

print ('sum of test data', df_test.isnull().sum())
#########################################

# prepare 2 temp data fram

# df_train_Non_Surived - data frame for all not survived

# df_train_Survived - data frame for all survived

df_train_Non_Survived = df_train.loc[(df_train.Survived == 0)]

df_train_Survived = df_train.loc[df_train.Survived == 1]

#df_train_Survived.describe()
##########################################

# 1. plot Fare vs Survived

##########################################



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#matplotlib.style.use('ggplot') 

#print(df_train['Fare'].values.max())



fig, axes = plt.subplots(nrows=2, ncols=2)



#hist for fare

ax1 = df_train.hist(column="Fare",        # Column to plot

              figsize=(8,8),         # Plot size

              color="blue",          # Plot color

              bins = 50, 

              ax=axes[0,0])

axes[0, 0].set_title('Fare Hist')



#box diagram - Fare by Survived

ax2 =df_train.boxplot(column="Fare",        # Column to plot

                 by= "Survived",       # Column to split upon

                 figsize= (8,8),       # Figure size

                 ax=axes[0,1])

ax2.set_ylim(-1, 125)

axes[0, 1].set_title('Fare by Survived')





#hist for fare when Survived = 0

ax3 = df_train_Non_Survived.hist(column="Fare",        # Column to plot

                                 figsize=(8,8),         # Plot size

                                 color="blue",          # Plot color

                                 bins = 50, 

                                 ax=axes[1,0])

axes[1, 0].set_title('Fare who Not Survived')



ax4 = df_train_Survived.hist(column="Fare",        # Column to plot

                             figsize=(8,8),         # Plot size

                             color="blue",          # Plot color

                             bins = 50,

                             ax = axes[1,1])

axes[1, 1].set_title('Fare who Survived')



#plt.show()
##########################################

# 2. plot Embarked vs Survived

##########################################



import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



fig, axes = plt.subplots(nrows=2, ncols=2)



#first plot: bar (freq) of Embarked

ax1 = df_train.Embarked.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[0,0])

axes[0, 0].set_title('Embarked Hist')



#second plot: Embarked vs Servived

sns.set(style="whitegrid", color_codes=True)

ax2 = sns.pointplot(x="Embarked", y="Survived", data=df_train, ax=axes[0,1]);

axes[0, 1].set_title('Embarked vs Survived')



#hist for fare when Survived = 0

ax3 = df_train_Non_Survived.Embarked.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,0])

axes[1, 0].set_title('Fare who Not Survived')



ax4 = df_train_Survived.Embarked.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,1])

axes[1, 1].set_title('Fare who Survived')
##########################################

# 3. plot Pclass vs Survived

##########################################



fig, axes = plt.subplots(nrows=2, ncols=2)



#first plot: bar (freq) of Pclass

ax1 = df_train.Pclass.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[0,0])

axes[0, 0].set_title('Pclass Hist')



#second plot: Pclass vs Servived

sns.set(style="whitegrid", color_codes=True)

#ax2 = sns.pointplot(x="Sex", y="Survived", data=df_train, ax=axes[0,1]);

ax2 = sns.barplot(x ="Pclass", y = "Survived", data=df_train, ax = axes[0,1])

axes[0, 1].set_title('Pclass vs Survived')



#hist for Pclass when Survived = 0

ax3 = df_train_Non_Survived.Pclass.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,0])

axes[1, 0].set_title('Pclass who Not Survived')



ax4 = df_train_Survived.Pclass.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,1])

axes[1, 1].set_title('Pclass who Survived')
##########################################

# 4. plot Sex vs Survived

##########################################



fig, axes = plt.subplots(nrows=2, ncols=2)



#first plot: bar (freq) of Gender

ax1 = df_train.Sex.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[0,0])

axes[0, 0].set_title('Gender Hist')



#second plot: Gender vs Servived

sns.set(style="whitegrid", color_codes=True)

ax2 = sns.barplot(x ="Sex", y = "Survived", data=df_train, ax = axes[0,1])

axes[0, 1].set_title('Gender vs Survived')



#hist for Sex when Survived = 0

ax3 = df_train_Non_Survived.Sex.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,0])

axes[1, 0].set_title('Pclass who Not Survived')



ax4 = df_train_Survived.Sex.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,1])

axes[1, 1].set_title('Pclass who Survived')
##########################################

# 5. plot Parch vs Survived

##########################################

#plt.tight_layout()

#plt.show()



fig, axes = plt.subplots(nrows=2, ncols=2)



#first plot: bar (freq) of Parch

ax1 = df_train.Parch.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[0,0])

axes[0, 0].set_title('Parch Hist')



#second plot: Parch vs Servived

sns.set(style="whitegrid", color_codes=True)

#ax2 = sns.pointplot(x="Sex", y="Survived", data=df_train, ax=axes[0,1]);

ax2 = sns.pointplot(x ="Parch", y = "Survived", data=df_train, ax = axes[0,1])

axes[0, 1].set_title('Parch vs Survived')



#hist for Parch when Survived = 0

ax3 = df_train_Non_Survived.Parch.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,0])

axes[1, 0].set_title('Parch who Not Survived')

#hist for Parch when Survived = 1

ax4 = df_train_Survived.Parch.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,1])

axes[1, 1].set_title('Parch who Survived')
##########################################

# 6. plot SibSp vs Survived

##########################################

fig, axes = plt.subplots(nrows=2, ncols=2)



#first plot: bar (freq) of SibSp

ax1 = df_train.SibSp.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[0,0])

axes[0, 0].set_title('SibSp Hist')



#second plot: SibSp vs Servived

sns.set(style="whitegrid", color_codes=True)

ax2 = sns.pointplot(x ="SibSp", y = "Survived", data=df_train, ax = axes[0,1])

axes[0, 1].set_title('SibSp vs Survived')



#hist for SibSp when Survived = 0

ax3 = df_train_Non_Survived.SibSp.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,0])

axes[1, 0].set_title('SibSp who Not Survived')

#hist for SibSp when Survived = 1

ax4 = df_train_Survived.SibSp.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,1])

axes[1, 1].set_title('SibSp who Survived')
##########################################

# 7. plot Age vs Survived

##########################################

fig, axes = plt.subplots(nrows=2, ncols=2)



#hist for Age

ax1 = df_train.hist(column="Age",        # Column to plot

              figsize=(8,8),         # Plot size

              color="blue",          # Plot color

              bins = 50, 

              ax=axes[0,0])

axes[0, 0].set_title('Age Hist')



#box diagram - Age by Survived

ax2 =df_train.boxplot(column="Age",        # Column to plot

                 by= "Survived",       # Column to split upon

                 figsize= (8,8),       # Figure size

                 ax=axes[0,1])

#ax2.set_ylim(-1, 125)

axes[0, 1].set_title('Age by Survived')





#hist for Age when Survived = 0

ax3 = df_train_Non_Survived.hist(column="Age",        # Column to plot

                                 figsize=(8,8),         # Plot size

                                 color="blue",          # Plot color

                                 bins = 50, 

                                 ax=axes[1,0])

axes[1, 0].set_title('Age who Not Survived')



ax4 = df_train_Survived.hist(column="Age",        # Column to plot

                             figsize=(8,8),         # Plot size

                             color="blue",          # Plot color

                             bins = 50,

                             ax = axes[1,1])

axes[1, 1].set_title('Age who Survived')
#################################################

# Get correlation between features

# Delete passengerID, not useful for Correlation

#################################################

df_train_clean = df_train.drop('PassengerId', 1)

print(df_train.describe())

print(df_train.corr())
######################################

#Age >65, P(Survived) --> 0

#Fare > 100, P(Surived) is increased

######################################

ax1 = df_train_Survived.plot.scatter(x='Age', y='Fare', c='r', s= 100, marker = '+');

ax2 = df_train_Non_Survived.plot.scatter(x='Age', y='Fare', c='b', s= 10,  marker = 'x', ax=ax1);

plt.show()
##################################################

# Imputing the missing data for age and embarked

##################################################

from sklearn.preprocessing import Imputer



#1 - train data

# mean - stragety for imput missing Age

tmp =df_train['Age'].values

tmp = tmp.reshape(-1,1)

imr = Imputer(missing_values = 'NaN', strategy = 'median', axis=0)

imr = imr.fit(tmp)

tmp = imr.transform(tmp)

df_train['Age'] = tmp



#most_frequent - strategy for imput missing Embarded

#Embarked is categorical data, mapping to Integer for imputting



embarked_mapping = {'S':3, 'C':2, 'Q':1}

inv_embarked_mapping = {v: k for k, v in embarked_mapping.items()}

df_train['Embarked']=df_train['Embarked'].map(embarked_mapping)



tmp =df_train['Embarked'].values

tmp = tmp.reshape(-1,1)

imr = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis=0)

imr = imr.fit(tmp)

tmp = imr.transform(tmp)

df_train['Embarked'] = tmp



df_train['Embarked']=df_train['Embarked'].map(inv_embarked_mapping)



#2 - test data

# mean - stragety for imput missing Age

tmp =df_test['Age'].values

tmp = tmp.reshape(-1,1)

imr = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)

imr = imr.fit(tmp)

tmp = imr.transform(tmp)

df_test['Age'] = tmp



# mean - stragety for imput missing Fare

tmp =df_test['Fare'].values

tmp = tmp.reshape(-1,1)

imr = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)

imr = imr.fit(tmp)

tmp = imr.transform(tmp)

df_test['Fare'] = tmp



#3 - show the summary of data

print ('sum of train data', df_train.isnull().sum())

print ('')

print ('sum of test data', df_test.isnull().sum())
test = df_train.tail()

print(test)



#Extract the data from the train.csv

#indx: contain the label of each columun

#X: contain all the passenager infor, except the label of 'Survived'

#y: the list of 'Survived'

index =[]

for i in range (len(df_train.axes[1])):

    if df_train.axes[1][i] != 'Survived':

        index.append(df_train.axes[1][i])      



       

print(index)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn import svm
df_tmp = df_train.drop('PassengerId',1)

df_tmp = df_tmp.drop('Survived', 1)



embarked_mapping = {'S':3, 'C':2, 'Q':1}

df_tmp['Embarked']=df_tmp['Embarked'].map(embarked_mapping)



Sex_mapping = {'male':0, 'female':1}

df_tmp['Sex']=df_tmp['Sex'].map(Sex_mapping)



feat_labels = df_tmp.columns[0:]

print(feat_labels)





X = df_tmp.values

y = df_train.Survived.values



forest = RandomForestClassifier(n_estimators=1000,

                               random_state=0,

                               n_jobs = -1)



forest.fit(X, y)

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]



print(indices)



plt.title('Feature Importance')

plt.bar (range(X.shape[1]),

         importances[indices],

         color = 'lightblue',

         align = 'center')

plt.xticks(range(X.shape[1]),

          feat_labels[indices], rotation = 90)
df_tmp = df_tmp.drop('Embarked', 1)

df_tmp = df_tmp.drop('SibSp', 1)

df_tmp = df_tmp.drop('Parch', 1)

#df_tmp = df_tmp.drop('Age', 1)





test = df_tmp.tail()

print(test)



X = df_tmp.values

y = df_train.Survived.values



clf1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)

scores = cross_val_score(clf1, X, y)

print(scores.mean())                             



clf2 = RandomForestClassifier(n_estimators=100,

                             criterion = 'entropy',

                             max_features = 2,

                             max_depth= 6,

                             min_samples_split=10, 

                             random_state=0)

scores = cross_val_score(clf2, X, y)

print (scores.mean())                             



clf3 = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)

scores = cross_val_score(clf3, X, y)

print (scores.mean())  



clf4 = svm.SVC(kernel = 'rbf',

          random_state = 0,

          gamma = 0.20,

          C=1.0)

scores = cross_val_score(clf4, X, y)

print (scores.mean())  



clf = clf4
clf = clf.fit(X, y)



pred = clf.predict(X)



from sklearn.metrics import confusion_matrix



CM = confusion_matrix(y, pred)



TN = CM[0][0]

FN = CM[1][0]

TP = CM[1][1]

FP = CM[0][1]



# Overall accuracy

ACC = (TP+TN)/(TP+FP+FN+TN)

print(TN, FN, TP, FP)

print(ACC)
#################################################

#Caluate predication from the training data

#################################################

#init pred vector

pred_test = []



df_test_tmp = df_test.drop('PassengerId',1)



embarked_mapping = {'S':3, 'C':2, 'Q':1}

df_test_tmp['Embarked']=df_test_tmp['Embarked'].map(embarked_mapping)



Sex_mapping = {'male':0, 'female':1}

df_test_tmp['Sex']=df_test_tmp['Sex'].map(Sex_mapping)



df_test_tmp=df_test_tmp.drop('Embarked', 1)

df_test_tmp = df_test_tmp.drop('SibSp', 1)

df_test_tmp = df_test_tmp.drop('Parch', 1)

#df_test_tmp = df_test_tmp.drop('Pclass', 1)                               



X_test = df_test_tmp.values

pred_test = clf.predict(X_test)
output = pd.DataFrame({ 'PassengerId': df_test["PassengerId"], "Survived": pred_test})

output.to_csv('prediction_05.csv', index=False)