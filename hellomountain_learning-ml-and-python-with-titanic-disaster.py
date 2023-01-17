#Config

%config IPCompleter.greedy=True #IPython autocomplete ON



#Libraries: data analysis and wrangling

import pandas as pd

import numpy as np

import graphviz



#Libraries: visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns





#Libraries: machine learning

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

from subprocess import call



#from sklearn.linear_model import LogisticRegression

#from sklearn.svm import SVC, LinearSVC

#from sklearn.ensemble import RandomForestClassifier

#from sklearn.neighbors import KNeighborsClassifier

#from sklearn.naive_bayes import GaussianNB

#from sklearn.linear_model import Perceptron

#from sklearn.linear_model import SGDClassifier

#from sklearn.model_selection import train_test_split

#Paths

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



main_path = '/kaggle/input/titanic/'

train_file = 'train.csv'

test_file = 'test.csv'



#Dataframe creations

df_train = pd.read_csv(main_path + train_file)

df_test = pd.read_csv(main_path + test_file)
#Explore the dimension

df_train.shape
#Explore features and lines

df_train.head(10)

#df_train.columns.values
#Explore features

df_train.info()

#Select null values

#df_train[df_train['Age'].isnull()]

#Explore categorical features

df_train.describe(include = 'object')
#Explore numerical features

df_train.describe()
#Select some data to have a better knowledge of the dataset. Examples:

#a)how is the data of the youngsters? 

#b)is the data from 'Cabin' useless?

#df_train.loc[df_train['Age']<20]

#df_train.loc[df_train['Cabin'].notnull()]
df_train2 = df_train

df_test2 = df_test
df_train2['_FamilySize'] = df_train2['SibSp'] + df_train2['Parch'] + 1

df_test2['_FamilySize'] = df_test2['SibSp'] + df_test2['Parch'] + 1
df_train2['_Title'] = df_train2['Name'].str.extract('([A-Za-z]+)\.')

df_test2['_Title'] = df_test2['Name'].str.extract('([A-Za-z]+)\.')
df_train2['_Cabin'] = pd.isnull(df_train2['Cabin'])

df_test2['_Cabin'] = pd.isnull(df_test2['Cabin'])
df_train2['_Age'] = df_train2.groupby(['_Title'])['Age'].apply(lambda x: x.fillna(x.median()))

df_test2['_Age'] = df_train2.groupby(['_Title'])['Age'].apply(lambda x: x.fillna(x.median()))
#Groups: 0,5,30,100

df_train2['_Age2'] = pd.cut(df_train2['_Age'], (0,5,30,100), labels=False )

df_test2['_Age2'] = pd.cut(df_test2['_Age'], (0,5,30,100), labels=False )

df_train2.head()
df_train2['_FamilySize'] = df_train2['_FamilySize'].replace([4, 5,6,7,8,9,10,11], 4 )

df_test2['_FamilySize'] = df_test2['_FamilySize'].replace([4, 5,6,7,8,9,10,11], 4 )

sns.countplot(x='Embarked',data=df_train2,palette='Set2')

plt.show()
freq_port = df_train2.Embarked.dropna().mode()[0]

df_train2['Embarked'] = df_train2['Embarked'].fillna(freq_port)

df_test2['Embarked'] = df_test2['Embarked'].fillna(freq_port)
df_train2_sex = pd.get_dummies(df_train2['Sex'])

df_test2_sex = pd.get_dummies(df_test2['Sex'])



df_train2 = pd.concat([df_train2, df_train2_sex], axis=1)

df_test2 = pd.concat([df_test2, df_test2_sex], axis=1)

df_train2
#df_train.loc[(df_train['Sex']=='male') & (df_train['Age']<15)]
#Create groups for titles

df_train2['_Title'] = df_train2['_Title'].replace(['Capt', 'Countess','Dr','Rev','Major','Col','Jonkheer','Mme'], 1)# Titles with distinction or rares

df_train2['_Title'] = df_train2['_Title'].replace(['Master','Mr','Sir','Don'], 2)# Males

df_train2['_Title'] = df_train2['_Title'].replace(['Mrs'], 3)# Married females

df_train2['_Title'] = df_train2['_Title'].replace(['Miss','Mlle','Ms','Dona','Lady'], 4)# Unmarried females

df_test2['_Title'] = df_test2['_Title'].replace(['Capt', 'Countess','Dr','Rev','Major','Col','Jonkheer','Mme'], 1)# Titles with distinction or rares

df_test2['_Title'] = df_test2['_Title'].replace(['Master','Mr','Sir','Don'], 2)# Males

df_test2['_Title'] = df_test2['_Title'].replace(['Mrs'], 3)# Married females

df_test2['_Title'] = df_test2['_Title'].replace(['Miss','Mlle','Ms','Dona','Lady'], 4)# Unmarried females



df_train2['Embarked'] = df_train2['Embarked'].replace(['S'], 1)

df_train2['Embarked'] = df_train2['Embarked'].replace(['C'], 2)

df_train2['Embarked'] = df_train2['Embarked'].replace(['Q'], 3)

df_test2['Embarked'] = df_test2['Embarked'].replace(['S'], 1)

df_test2['Embarked'] = df_test2['Embarked'].replace(['C'], 2)

df_test2['Embarked'] = df_test2['Embarked'].replace(['Q'], 3)
df_train2 = df_train2.drop(['SibSp','Age','_Age','Parch','Ticket','Fare','Name','Cabin','Sex','female'],axis=1)

df_test2 = df_test2.drop(['SibSp','Age','_Age','Parch','Ticket','Fare','Name','Cabin','Sex','female'],axis=1)
df_train2.info()

#df_test2.info()
df_train2.head()

#df_test2.head()
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation', y=1, size=15)

sns.heatmap(df_train2.astype(float).corr(),linewidths=0.1,vmax=1.0, cmap=colormap, linecolor='white', annot=True)
df_train2.corr()['Survived']
group_Sex = df_train2.groupby('male').agg({'PassengerId': 'count','Survived': 'mean'}) 

group_Sex 

#Result: high probability (74%) that woman survive
group_Pclass = df_train2.groupby('Pclass').agg({'PassengerId': 'count','Survived': 'mean'}) 

group_Pclass

#Result: high probability (76%) that passengers in class 3 does not survive
group_FamilySize = df_train2.groupby('_FamilySize').agg({'PassengerId': 'count', 'Survived': 'mean'})

group_FamilySize

#Result: very high probability (94%) that passengers with big family (+3) does not survive and high probability (70%) that passengers travelling alone does not survive
group_cabin = df_train2.groupby('_Cabin').agg({'PassengerId': 'count', 'Survived': 'mean'})

group_cabin

#Result: Cabin does not help to survive (only 29%)
group_embarked = df_train2.groupby('Embarked').agg({'PassengerId': 'count', 'Survived': 'mean'})

group_embarked

#Result: Passengers embarked in C have more possibilities to survive
df_train2['_Age2'].hist(by=df_train['Survived'])

#Results: for age, distributions are similar except for groups lower than 15 and greater than 30 where it seems that they have more posibilities to survive
group_title = df_train2.groupby('_Title').agg({'PassengerId': 'count', 'Survived': 'mean'})

group_title

#Result: reverends do not survive! and there is no big difference between married/unmarried women
df_train2.head()
X_train = df_train2.drop(['PassengerId','Survived','male','_Cabin'],axis=1)

Y_train = df_train2['Survived']

X_test = df_test2.drop(['PassengerId','male','_Cabin'],axis=1)
DecisionTree = DecisionTreeClassifier(criterion='entropy', splitter='best',max_depth=4, min_samples_leaf=10)

DecisionTree.fit(X_train, Y_train)

Y_pred = DecisionTree.predict(X_test)

ResultDecisionTree = DecisionTree.score(X_train, Y_train)

ResultDecisionTree
#Let's see our tree

tree1_view = tree.export_graphviz(DecisionTree, out_file=None, feature_names = X_train.columns.values, rotate=True, class_names = ['Died', 'Survived']) 

tree1viz = graphviz.Source(tree1_view)

tree1viz
submission = pd.DataFrame({

        "PassengerId": df_test2["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic2.csv', index=False)
