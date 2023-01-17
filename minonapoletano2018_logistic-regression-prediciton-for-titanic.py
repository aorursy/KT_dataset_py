# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools

import sys

import seaborn



sns.set(style='white',context='notebook',palette='deep')





%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.

seaborn.set() 

dataset= pd.read_csv("../input/train.csv")



survived_class = dataset[dataset['Survived']==1]['Pclass'].value_counts()

dead_class = dataset[dataset['Survived']==0]['Pclass'].value_counts()

df_class = pd.DataFrame([survived_class,dead_class])

df_class.index = ['Survived','Died']

df_class.plot(kind='bar',stacked=True, figsize=(5,3), title="Survived/Died by Class")



Class1_survived= df_class.iloc[0,0]/df_class.iloc[:,0].sum()*100

Class2_survived = df_class.iloc[0,1]/df_class.iloc[:,1].sum()*100

Class3_survived = df_class.iloc[0,2]/df_class.iloc[:,2].sum()*100

print("Percentage of Class 1 that survived:" ,round(Class1_survived),"%")

print("Percentage of Class 2 that survived:" ,round(Class2_survived), "%")

print("Percentage of Class 3 that survived:" ,round(Class3_survived), "%")



# display table

from IPython.display import display

display(df_class)



#-------------------Survived/Died by SEX------------------------------------

Survived = dataset[dataset.Survived == 1]['Sex'].value_counts()

Died = dataset[dataset.Survived == 0]['Sex'].value_counts()

df_sex = pd.DataFrame([Survived , Died])

df_sex.index = ['Survived','Died']

df_sex.plot(kind='bar',stacked=True, figsize=(5,3), title="Survived/Died by Sex")





female_survived= df_sex.female[0]/df_sex.female.sum()*100

male_survived = df_sex.male[0]/df_sex.male.sum()*100

print("Percentage of female that survived:" ,round(female_survived), "%")

print("Percentage of male that survived:" ,round(male_survived), "%")



# display table

from IPython.display import display

display(df_sex) 



#-------------------- Survived/Died by Embarked ----------------------------



survived_embark = dataset[dataset['Survived']==1]['Embarked'].value_counts()

dead_embark = dataset[dataset['Survived']==0]['Embarked'].value_counts()

df_embark = pd.DataFrame([survived_embark,dead_embark])

df_embark.index = ['Survived','Died']

df_embark.plot(kind='bar',stacked=True, figsize=(5,3))



Embark_S= df_embark.iloc[0,0]/df_embark.iloc[:,0].sum()*100

Embark_C = df_embark.iloc[0,1]/df_embark.iloc[:,1].sum()*100

Embark_Q = df_embark.iloc[0,2]/df_embark.iloc[:,2].sum()*100

print("Percentage of Embark S that survived:", round(Embark_S), "%")

print("Percentage of Embark C that survived:" ,round(Embark_C), "%")

print("Percentage of Embark Q that survived:" ,round(Embark_Q), "%")



from IPython.display import display

display(df_embark)

X = dataset.drop(['PassengerId','Cabin','Ticket','Fare', 'Parch', 'SibSp'], axis=1)

y = X.Survived                       # vector of labels (dependent variable)

X=X.drop(['Survived'], axis=1)       # remove the dependent variable from the dataframe X



X.head(20)





test = pd.read_csv('../input/test.csv')



# -------- Change Name -> Title ----------------------------

got= test.Name.str.split(',').str[1]

test.iloc[:,2]=pd.DataFrame(got).Name.str.split('\s+').str[1]

# ---------------------------------------------------------- 

test = test.drop(['PassengerId', 'Ticket', 'Fare', 'Cabin', 'Parch', 'SibSp'], axis=1)



#------------------ Average Age per title -------------------------------------------------------------

ax = plt.subplot()

ax.set_ylabel('Average age')

test.groupby('Name').mean()['Age'].plot(kind='bar',figsize=(13,8), ax = ax)



title_mean_age=[]

title_mean_age.append(list(set(test.Name)))  #set for unique values of the title, and transform into list

title_mean_age.append(test.groupby('Name').Age.mean())

title_mean_age



#------------------------------------------------------------------------------------------------------





#------------------ Fill the missing Ages ---------------------------

n_traning= test.shape[0]   #number of rows

n_titles= len(title_mean_age[1])

for i in range(0, n_traning):

    if np.isnan(test.Age[i])==True:

        for j in range(0, n_titles):

            if test.Name[i] == title_mean_age[0][j]:

                test.Age[i] = title_mean_age[1][j]

#--------------------------------------------------------------------    



test=test.drop(['Name'], axis=1)



for i in range(0, n_traning):

    if test.Age[i] > 18:

        test.Age[i]= 0

    else:

        test.Age[i]= 1

# encode "Sex"

from sklearn.preprocessing import LabelEncoder

labelEncoder_test = LabelEncoder()

test.Sex=labelEncoder_test.fit_transform(test.Sex)



# encode "Embarked"



# fill the two values with one of the options (S, C or Q)

row_index = test.Embarked.isnull()

test.loc[row_index,'Embarked']='S' 



Embarked  = pd.get_dummies(  test.Embarked , prefix='Embarked'  )

test = test.drop(['Embarked'], axis=1)

test= pd.concat([test, Embarked], axis=1)  

# we should drop one of the columns

test = test.drop(['Embarked_S'], axis=1)



# ----------------- Encoding categorical data -------------------------



labelEncoder_X = LabelEncoder()

X.Sex=labelEncoder_X.fit_transform(X.Sex)



# encode "Embarked"



# number of null values in embarked:

print ('Number of null values in Embarked:', sum(X.Embarked.isnull()))



# fill the two values with one of the options (S, C or Q)

row_index = X.Embarked.isnull()

X.loc[row_index,'Embarked']='S' 



Embarked  = pd.get_dummies(  X.Embarked , prefix='Embarked'  )

X = X.drop(['Embarked'], axis=1)

X= pd.concat([X, Embarked], axis=1)  

# we should drop one of the columns

X = X.drop(['Embarked_S'], axis=1)



X.head()

#-------------- Taking care of missing data  -----------------------------



print ('Number of null values in Age:', sum(X.Age.isnull()))

 



# -------- Change Name -> Title ----------------------------

got= dataset.Name.str.split(',').str[1]

X.iloc[:,1]=pd.DataFrame(got).Name.str.split('\s+').str[1]

# ---------------------------------------------------------- 





#------------------ Average Age per title -------------------------------------------------------------

ax = plt.subplot()

ax.set_ylabel('Average age')

X.groupby('Name').mean()['Age'].plot(kind='bar',figsize=(13,8), ax = ax)



title_mean_age=[]

title_mean_age.append(list(set(X.Name)))  #set for unique values of the title, and transform into list

title_mean_age.append(X.groupby('Name').Age.mean())

title_mean_age



#------------------------------------------------------------------------------------------------------





#------------------ Fill the missing Ages ---------------------------

n_traning= dataset.shape[0]   #number of rows

n_titles= len(title_mean_age[1])

for i in range(0, n_traning):

    if np.isnan(X.Age[i])==True:

        for j in range(0, n_titles):

            if X.Name[i] == title_mean_age[0][j]:

                X.Age[i] = title_mean_age[1][j]

#--------------------------------------------------------------------    



X=X.drop(['Name'], axis=1)



for i in range(0, n_traning):

    if X.Age[i] > 18:

        X.Age[i]= 0

    else:

        X.Age[i]= 1



X.head()





#-----------------------Logistic Regression---------------------------------------------

# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(penalty='l2',random_state = 0)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)

print("Logistic Regression:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std(),"\n")







#-----------------------------------K-NN --------------------------------------------------



# Fitting K-NN to the Training set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)





# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)

print("K-NN:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std(),"\n")





#---------------------------------------SVM -------------------------------------------------



# Fitting Kernel SVM to the Training set

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)

print("SVM:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std(),"\n")



#---------------------------------Naive Bayes-------------------------------------------



# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)

print("Naive Bayes:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std(),"\n")







#----------------------------Random Forest------------------------------------------



# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)

print("Random Forest:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std())



classifier.fit(X, y)



pre = classifier.predict(test)



test1 = pd.read_csv('../input/test.csv')



pre_pd = pd.DataFrame(test1['PassengerId'])



pre_pd['Survived'] =  pd.Series(pre)



pre_pd.to_csv("my_titanicsubmission.csv",index=False)





