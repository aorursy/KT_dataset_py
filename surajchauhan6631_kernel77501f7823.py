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
import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
dataset = pd.read_csv("../input/train.csv")
dataset.shape
dataset.head()
import seaborn

seaborn.set()
survived_class = dataset[dataset['Survived']==1]['Pclass'].value_counts()

dead_class = dataset[dataset['Survived']==0]['Pclass'].value_counts()

df_class = pd.DataFrame([survived_class,dead_class])

df_class.index = ['Survived','Dead']

df_class.plot(kind='bar',stacked=True,figsize=(5,3),title='Survivied/Dead by class')



Class1_survived = df_class.iloc[0,0]/df_class.iloc[:,0].sum()*100

Class2_survived = df_class.iloc[0,1]/df_class.iloc[:,1].sum()*100

Class3_survived = df_class.iloc[0,2]/df_class.iloc[:,2].sum()*100



print("Percentage of class1 that survived ",Class1_survived)

print("Percentage of class2 that survived ",Class2_survived)

print("Percentage of class3 that survived ",Class3_survived)



from IPython.display import display

display(df_class)
Survived = dataset[dataset.Survived==1]['Sex'].value_counts()

dead = dataset[dataset.Survived==0]['Sex'].value_counts()

df_sex = pd.DataFrame([Survived,dead])

df_sex.index = ['survived','dead']

df_sex.plot(kind='bar',stacked=True,figsize=(5,3),title='survived/dead by sex')

female_survived = df_sex.female[0]/df_sex.female.sum()*100

male_survivied = df_sex.male[0]/df_sex.male.sum()*100



print("Percentage of male survived ",male_survivied)

print("Percentage of female survived ",female_survived)

display(df_sex)

## Survived/Died by embarked

survived_embark = dataset[dataset['Survived']==1]['Embarked'].value_counts()

dead_embark = dataset[dataset.Survived==0]['Embarked'].value_counts()

df_embark = pd.DataFrame([survived_embark,dead_embark])

df_embark.index = ['Survived','Died']

df_embark.plot(kind='bar',stacked=True,figsize=(5,3),title='survived/dead by embark')



embark_S = df_embark.iloc[0,0]/df_embark.iloc[:,0].sum()*100

embark_C = df_embark.iloc[0,1]/df_embark.iloc[:,1].sum()*100

embark_Q = df_embark.iloc[0,2]/df_embark.iloc[:,2].sum()*100



print("Percentage of embark s that survived ",embark_S)

print("Percentage of embark c that survived ",embark_C)

print("Percentage of embark q that survived ",embark_Q)

display(df_embark)
dataset.columns
X = dataset.drop(['PassengerId','Cabin','Ticket','Fare','Parch','SibSp'],axis=1)

y = X.Survived

X = X.drop(['Survived'],axis=1)

X.head()
from sklearn.preprocessing import LabelEncoder

LabelEncoder_X = LabelEncoder()

X.Sex = LabelEncoder_X.fit_transform(X.Sex)



print("No of null values in embarked ",sum(X.Embarked.isnull()))

row_index = X.Embarked.isnull()

X.loc[row_index,'Embarked']='S'



Embarked = pd.get_dummies(X.Embarked,prefix='Embarked')

X = X.drop(['Embarked'],axis=1)

X = pd.concat([X,Embarked],axis=1)

X.head()
# Taking care of missing data

print("Number of NULL values in age: ",X.Age.isnull().sum())

got = dataset.Name.str.split(',').str[1]

X.iloc[:,1] = pd.DataFrame(got).Name.str.split('\s+').str[1]
# ---------------average age per title -------------------

ax = plt.subplot()

ax.set_ylabel('Average age')

X.groupby('Name').mean()['Age'].plot(kind='bar',figsize=(10,13),ax=ax)

title_mean_age =[]

title_mean_age.append(list(set(X.Name)))

title_mean_age.append(X.groupby('Name').Age.mean())

print(title_mean_age)
## Filling the missing age

n_training = dataset.shape[0]

n_titles = len(title_mean_age[1])

for i in range(0,n_training):

    if np.isnan(X.Age[i])==True:

        for j in range(0,n_titles):

            if X.Name[i]==title_mean_age[0][j]:

                X.Age[i] =title_mean_age[1][j]

                

X = X.drop(['Name'],axis=1)
for i in range(0,n_training):

    if X.Age[i] > 18:

        X.Age[i]=0

    else:

        X.Age[i]=1

X.head()
#---------------Logistic Regression-----------------

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(penalty='l2',random_state=0)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier,X=X,y=y,cv=10)

print('Logistic Regression:\n Accuracy: ',accuracies.mean(),'+/-',accuracies.std())



# --------------------K-NN----------------

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=9,metric='minkowski',p=2)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier,X=X,y=y,cv=10)

print('K-NN :\n Accuracy: ',accuracies.mean(),'+/-',accuracies.std())



#------------------SVM-----------------------------------

from sklearn.svm import SVC

classifier = SVC(kernel='rbf',random_state=0)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier,X=X,y=y,cv=10)

print('SVM:\n Accuracy: ',accuracies.mean(),'+/-',accuracies.std())



#---------------------------- Naive Bayes -----------------------

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier,X=X,y=y,cv=10)

print('GaussianNB :\n Accuracy: ',accuracies.mean(),'+/-',accuracies.std())



#---------------------------------Random Forest--------------------------------

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier,X=X,y=y,cv=10)

print('Random Forest:\n Accuracy: ',accuracies.mean(),'+/-',accuracies.std())
test_data = pd.read_csv('../input/test.csv')
test_data.head()