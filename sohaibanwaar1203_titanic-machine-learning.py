# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sp #collection of functions for scientific computing and advance mathematics

print("SciPy version: {}". format(sp.__version__)) 



import IPython

from IPython import display #pretty printing of dataframes in Jupyter notebook

print("IPython version: {}". format(IPython.__version__)) 



import sklearn #collection of machine learning algorithms

print("scikit-learn version: {}". format(sklearn.__version__))

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import train_test_split

from sklearn.linear_model import PassiveAggressiveClassifier,RidgeClassifierCV,SGDClassifier,Perceptron





import os

print(os.listdir("../input"))

from keras.utils import to_categorical

# Any results you write to the current directory are saved as output.
df_train=pd.read_csv("../input/train.csv")

df_test=pd.read_csv("../input/test.csv")



print('Train columns with null values:\n', df_train.isnull().sum())

print("-"*10)



print('Test/Validation columns with null values:\n',df_test.isnull().sum())

print("-"*10)



df_train.describe(include = 'all')


#we didnt need passenger id I am droppring it.

#for now we didnt need names too for machine learning

#we didnt need ticket number

df_train=df_train.drop("PassengerId",axis=1)

df_train=df_train.drop("Name",axis=1)

df_train=df_train.drop("Ticket",axis=1)

# male=1 and female=0 making my dataset numeric

df_train.loc[df_train['Sex'] =="male", 'Sex'] = 1

df_train.loc[df_train['Sex'] =="female", 'Sex'] = 0

#C = Cherbourg=1, Q = Queenstown=2, S = Southampton=3

df_train.loc[df_train['Embarked'] =="C", 'Embarked'] =1

df_train.loc[df_train['Embarked'] =="Q", 'Embarked'] =2

df_train.loc[df_train['Embarked'] =="S", 'Embarked'] =3

#2 missing values in Embarked coloum 

print("Coloum Embarked has missing values= ",df_train["Embarked"].isnull().values.sum())

print("Coloum Cabin has missing values= ",df_train["Cabin"].isnull().values.sum())

print("Coloum Age has missing values= ",df_train["Age"].isnull().values.sum())

print("At first we wil see co-relation of these coloums than we decide we have to drop these coloums or not")

df_train.Embarked.fillna(0, inplace=True)

df_train.head()
sns.heatmap(df_train.isnull())

#cabin has to much missing values so its good to grop it.

df_train=df_train.drop("Cabin",axis=1)

df_train=df_train.drop("Age",axis=1)





sns.heatmap(df_train.isnull())
df_train.groupby(['Parch'])['Survived'].count()

df_train.groupby(['SibSp'])['Survived'].count()
corr = df_train.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(df_train.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(df_train.columns)

ax.set_yticklabels(df_train.columns)

plt.show()
sns.pairplot(df_train, hue="Survived")
#here fare and pclass representing the same thing. increase the fare and more luxury seat you get 1st class etc

df_train=df_train.drop("Fare",axis=1)
df_train.hist(figsize=(10,8),bins=10,color='#ffd700',linewidth='1',edgecolor='k')

plt.tight_layout()

plt.show()
fig = plt.figure(figsize = (20, 25))

j = 0

#Droping_Characters and string coloums because graph donot support them



for i in df_train.columns:

    plt.subplot(6, 4, j+1)

    j += 1

    sns.distplot(df_train[i][df_train['Survived']==1], color='g', label = 'Survived')

    sns.distplot(df_train[i][df_train['Survived']==0], color='r', label = 'Not Survived')

    plt.legend(loc='best')

fig.suptitle('Admission Chance In University ')

fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
#outliers

for column in df_train:

    plt.figure()

    sns.boxplot(x=df_train[column])
df_train.columns
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot("Sex","Pclass",hue="Survived", data=df_train,split=True)

plt.subplot(2,2,2)

sns.violinplot("Sex","SibSp",hue="Survived", data=df_train,split=True)

plt.subplot(2,2,3)

sns.violinplot("Sex","Parch",hue="Survived", data=df_train,split=True)

plt.subplot(2,2,4)

sns.violinplot("Sex","Embarked",hue="Survived", data=df_train,split=True)

plt.ioff()

plt.show()
X=df_train.drop("Survived",axis=1)

y=df_train["Survived"]





print(X.shape)

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(

   X,y, test_size=0.1, random_state=0)



type(X_train)

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)



results=cross_val_score(clf,X_train ,y_train , cv=20)

print(results)

Average=sum(results) / len(results) 

print("Average Accuracy :",Average)
clf.fit(X_train,y_train)
feature_names = X.columns

importance_frame = pd.DataFrame()

importance_frame['Features'] = X.columns

importance_frame['Importance'] = clf.feature_importances_

importance_frame = importance_frame.sort_values(by=['Importance'], ascending=True)

plt.barh([1,2,3,4,5], importance_frame['Importance'], align='center', alpha=0.5)

plt.yticks([1,2,3,4,5], importance_frame['Features'])

plt.xlabel('Importance')

plt.title('Feature Importances')

plt.show()
df_train=pd.read_csv("../input/train.csv")

df_test=pd.read_csv("../input/test.csv")

#we didnt need passenger id I am droppring it.

#for now we didnt need names too for machine learning



df_train=df_train.drop("PassengerId",axis=1)

df_train=df_train.drop("Name",axis=1)



# male=1 and female=0 making my dataset numeric

df_train.loc[df_train['Sex'] =="male", 'Sex'] = 1

df_train.loc[df_train['Sex'] =="female", 'Sex'] = 0

#C = Cherbourg=1, Q = Queenstown=2, S = Southampton=3

df_train.loc[df_train['Embarked'] =="C", 'Embarked'] =1

df_train.loc[df_train['Embarked'] =="Q", 'Embarked'] =2

df_train.loc[df_train['Embarked'] =="S", 'Embarked'] =3

#2 missing values in Embarked coloum 

print("Coloum Embarked has missing values= ",df_train["Embarked"].isnull().values.sum())

print("Coloum Cabin has missing values= ",df_train["Cabin"].isnull().values.sum())

print("Coloum Age has missing values= ",df_train["Age"].isnull().values.sum())

print("At first we wil see co-relation of these coloums than we decide we have to drop these coloums or not")

#filling with S = Southampton=3 because most of the records are from S = Southampton=3

df_train.Embarked.fillna(3, inplace=True)

df_train.Cabin.fillna("0",inplace=True)



print("Unique values in Tickets",len(df_train["Ticket"].unique()))



df_train.head()



# now we know that numbers of Ticket represent useful information like in which part are you sitting

# in which area. Now people who are sitting in the middle are its very difficult for them to get our

# of boat and survive so we try to extract some useful information from this lets see



len(df_train["Cabin"].unique())

for i in range (0,len(df_train["Ticket"])):

    df_train["Ticket"].iloc[i]=df_train["Ticket"].iloc[i][:5]

    df_train["Cabin"].iloc[i]=df_train["Cabin"].iloc[i][:1]

    

    
# we are taking only first 5 digits that may be representing something useful that in which part these seats are 

#located etc.

len(df_train["Ticket"].unique())

df_train["Ticket"] = df_train["Ticket"].astype('category')

df_train["Ticket"]=df_train["Ticket"].cat.codes

df_train["Cabin"] = df_train["Cabin"].astype('category')

df_train["Cabin"]=df_train["Cabin"].cat.codes
#filling age missing values with average

df_train["Age"]=df_train["Age"].fillna(df_train["Age"].mean())
df_train["Cabin"]=df_train["Cabin"].replace(2,1)

df_train["Cabin"]=df_train["Cabin"].replace(3,1)

df_train["Cabin"]=df_train["Cabin"].replace(4,1)

df_train["Cabin"]=df_train["Cabin"].replace(5,1)

df_train["Cabin"]=df_train["Cabin"].replace(6,1)

df_train["Cabin"]=df_train["Cabin"].replace(7,1)

df_train["Cabin"]=df_train["Cabin"].replace(8,1)
sns.heatmap(df_train.isnull())
corr = df_train.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(df_train.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(df_train.columns)

ax.set_yticklabels(df_train.columns)

plt.show()
#df_train=df_train.drop("Fare",axis=1)



X=df_train.drop("Survived",axis=1)

y=df_train["Survived"]





print(X.shape)

print(y.shape)

print(type(X))

print(type(y))
X_train, X_test, y_train, y_test = train_test_split(

   X,y, test_size=0.1, random_state=0)



type(X_train)



from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)



result=cross_val_score(clf,X_train ,y_train , cv=10)

print(result)

Average=sum(result) / len(result) 

print("Average Accuracy :",Average)
clf.fit(X_train,y_train)
feature_names = X.columns

importance_frame = pd.DataFrame()

importance_frame['Features'] = X.columns

importance_frame['Importance'] = clf.feature_importances_

importance_frame = importance_frame.sort_values(by=['Importance'], ascending=True)

plt.barh([1,2,3,4,5,6,7,8,9], importance_frame['Importance'], align='center', alpha=0.5)

plt.yticks([1,2,3,4,5,6,7,8,9], importance_frame['Features'])

plt.xlabel('Importance')

plt.title('Feature Importances')

plt.show()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import numpy as np

from sklearn.model_selection import train_test_split

import keras

from keras.callbacks import LambdaCallback

from keras.layers import Conv1D, Flatten

from keras.layers import Dense ,Dropout,BatchNormalization

from keras.models import Sequential 

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical 

from keras import regularizers

from sklearn import preprocessing

from sklearn.ensemble import  VotingClassifier

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../working"))

from sklearn import ensemble

from sklearn import gaussian_process

from sklearn import linear_model

from sklearn import naive_bayes

from sklearn import neighbors

from sklearn import svm

from sklearn import tree

from sklearn import discriminant_analysis

from sklearn import model_selection

from xgboost.sklearn import XGBClassifier 
#Machine Learning Algorithm (MLA) Selection and Initialization

MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    XGBClassifier()    

    ]









#create table to compare MLA metrics

MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Test Accuracy' ]

MLA_compare = pd.DataFrame(columns = MLA_columns)







#index through MLA and save performance to table

row_index = 0

for alg in MLA:



    #set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate

   # cv_results = model_selection.cross_validate(alg, X_train, y_train)

    alg.fit(X_train, y_train)

    y_pred=alg.predict(X_test)

    score=metrics.accuracy_score(y_test, y_pred)

    

    MLA_compare.loc[row_index, 'MLA Test Accuracy'] =score



    

    

    row_index+=1



    



MLA_compare

#MLA_predict
X_train, X_test, y_train, y_test = train_test_split(

   X,y, test_size=0.1, random_state=0)
vote_clf = VotingClassifier(estimators=[

                                        

                                        ("DT",tree.DecisionTreeClassifier()),

                                        ("AD",ensemble.AdaBoostClassifier()),

                                        ("XGB",XGBClassifier()) ,

                                        ("GNB",naive_bayes.GaussianNB()),

                                        

    

    

                                        ])

vote_clf = vote_clf.fit(X_train, y_train)
result=cross_val_score(vote_clf,X_train ,y_train , cv=10)

print(result)

Average=sum(result) / len(result) 

print("Average Accuracy :",Average)
y_pred=vote_clf .predict(X_test)

print("Accuracy of : ",i,"\n",metrics.accuracy_score(y_test, y_pred))

   
clf=svm.NuSVC(probability=True)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
vote_clf = vote_clf.fit(X, y)
df_train=pd.read_csv("../input/train.csv")

df_test=pd.read_csv("../input/test.csv")

#we didnt need passenger id I am droppring it.

#for now we didnt need names too for machine learning



df_test=df_test.drop("PassengerId",axis=1)

df_test=df_test.drop("Name",axis=1)



# male=1 and female=0 making my dataset numeric

df_test.loc[df_test['Sex'] =="male", 'Sex'] = 1

df_test.loc[df_test['Sex'] =="female", 'Sex'] = 0

#C = Cherbourg=1, Q = Queenstown=2, S = Southampton=3

df_test.loc[df_test['Embarked'] =="C", 'Embarked'] =1

df_test.loc[df_test['Embarked'] =="Q", 'Embarked'] =2

df_test.loc[df_test['Embarked'] =="S", 'Embarked'] =3

#2 missing values in Embarked coloum 

print("Coloum Embarked has missing values= ",df_test["Embarked"].isnull().values.sum())

print("Coloum Cabin has missing values= ",df_test["Cabin"].isnull().values.sum())

print("Coloum Age has missing values= ",df_test["Age"].isnull().values.sum())

print("At first we wil see co-relation of these coloums than we decide we have to drop these coloums or not")

#filling with S = Southampton=3 because most of the records are from S = Southampton=3

df_test.Embarked.fillna(3, inplace=True)

df_test.Cabin.fillna("0",inplace=True)



print("Unique values in Tickets",len(df_test["Ticket"].unique()))



df_test.head()



# now we know that numbers of Ticket represent useful information like in which part are you sitting

# in which area. Now people who are sitting in the middle are its very difficult for them to get our

# of boat and survive so we try to extract some useful information from this lets see



for i in range (0,len(df_test["Ticket"])):

    df_test["Ticket"].iloc[i]=df_test["Ticket"].iloc[i][:5]

    df_test["Cabin"].iloc[i]=df_test["Cabin"].iloc[i][:1]

    

    
len(df_test["Ticket"].unique())

df_test["Ticket"] = df_test["Ticket"].astype('category')

df_test["Ticket"]=df_test["Ticket"].cat.codes

df_test["Cabin"] = df_test["Cabin"].astype('category')

df_test["Cabin"]=df_test["Cabin"].cat.codes
df_test["Age"]=df_test["Age"].fillna(df_test["Age"].mean())

df_test["Fare"]=df_test["Fare"].fillna(df_test["Fare"].mean())
df_test["Cabin"]=df_test["Cabin"].replace(2,1)

df_test["Cabin"]=df_test["Cabin"].replace(3,1)

df_test["Cabin"]=df_test["Cabin"].replace(4,1)

df_test["Cabin"]=df_test["Cabin"].replace(5,1)

df_test["Cabin"]=df_test["Cabin"].replace(6,1)

df_test["Cabin"]=df_test["Cabin"].replace(7,1)

df_test.head(10)
df_test.isnull().sum()
y_pred=vote_clf.predict(df_test)

test=pd.read_csv("../input/test.csv")
vote_submit = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })

vote_submit.to_csv("vote.csv", index=False)