import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import LabelEncoder



# spliting the data into two part train and test 

from sklearn.model_selection import train_test_split



# import the Requried Algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style

import seaborn as sn



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv("/kaggle/input/titanic/train.csv")

print(df_train.head())



df_test=pd.read_csv("/kaggle/input/titanic/test.csv")

df_test.head()
print(df_train.shape)

print(df_test.shape)
df_train.info()
df_test.isnull().sum()
# All calculating summary of our dataset.

df_train.describe()
df_train.isnull().sum()
count_sex = pd.value_counts(df_train['Sex'], sort = True)



count_sex.plot(kind = 'bar', rot=0)



plt.title("Gender Count")



plt.xticks(range(2))



plt.xlabel("Sex")



plt.ylabel("Count")
plt.scatter(df_train.Survived,df_train.Age,alpha=0.5)

plt.title("Age wrt Survived")
df_train.Pclass.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.show()
df_train.Embarked.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.show()
# count in the %age based on the survived male and female.

sn.barplot(x="Sex",y='Survived',data=df_train,palette='rainbow')

print("Male",df_train['Survived'][df_train['Sex']=='male'].value_counts(normalize=True)[1]*100)

print('Female',df_train['Survived'][df_train['Sex']=='female'].value_counts(normalize=True)[1]*100)
df_train.Survived[(df_train.Sex=='male') & (df_train.Pclass==1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Survived Rich Man")
df_train.Survived[(df_train.Sex=='male') & (df_train.Pclass==3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Survived Poor man")

plt.show()
df_train.Survived[(df_train.Sex=='female') & (df_train.Pclass==1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Survived Rich woman")
df_train.Survived[(df_train.Sex=='female') & (df_train.Pclass==3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Survived Poor woman")
sn.set_style('whitegrid')

sn.countplot(df_train['Survived'],hue='Pclass',data=df_train,palette='rainbow')
# sibsp: #of siblings/spouses aboard the Titanic

sn.countplot(df_train['SibSp'],data=df_train)
# parch:# of parents/children aboard the Titanic

sn.countplot(df_train['Parch'],data=df_train)
def data_clean(df_train):

    df_train["Fare"]=df_train['Fare'].fillna(df_train['Fare'].dropna().median())

    df_train["Age"]=df_train['Age'].fillna(df_train['Age'].dropna().median())

    

    df_train.loc[df_train['Sex']=='male','Sex']=1

    df_train.loc[df_train['Sex']=='female','Sex']=0

    

    df_train["Embarked"]=df_tran['Embarked'].fillna('S')

    df_train.loc[df_train['Embarked']=="S","Embarked"]=0

    df_train.loc[df_train['Embarked']=="C","Embarked"]=1

    df_train.loc[df_train['Embarked']=="Q","Embarked"]=2

    
df_train.head(2)
df_train.loc[df_train['Sex']=='male','Sex']=1

df_train.loc[df_train['Sex']=='female','Sex']=0
df_train["Age"]=df_train['Age'].fillna(df_train['Age'].dropna().median())
df_train["Embarked"]=df_train['Embarked'].fillna('S')

df_train.loc[df_train['Embarked']=="S","Embarked"]=0

df_train.loc[df_train['Embarked']=="C","Embarked"]=1

df_train.loc[df_train['Embarked']=="Q","Embarked"]=2
df_train=df_train.drop(['Cabin'],axis=1)
# Features

X = df_train[["Pclass","Age",'Sex','SibSp','Parch']].values

# Target

y = df_train['Survived'].values
X.shape
classifier= DecisionTreeClassifier()

classifier_ = classifier.fit(X,y)
print(classifier_.score(X,y))
# Features

X = df_train[["Pclass","Age",'Sex','Embarked','SibSp','Parch']].values

# Target

y = df_train['Survived'].values
classifier= DecisionTreeClassifier()

classifier_ = classifier.fit(X,y)
print(classifier_.score(X,y))
rfc=RandomForestClassifier()

random_clf=rfc.fit(X,y)
print(random_clf.score(X,y))
# Features

X = df_train[["Pclass","Age",'Sex','SibSp','Parch']].values

# Target

y = df_train['Survived'].values
rfc=RandomForestClassifier()

random_clf=rfc.fit(X,y)
print(random_clf.score(X,y))
# Fit logistic regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg_=logreg.fit(X, y)
print(logreg_.score(X,y))
from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV
# Choose some parameter combinations to try

parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(rfc, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X, y)



# Set the random forest clf to the best combination of parameters

rfc = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

rfc.fit(X, y)
clf=RandomForestClassifier(criterion='entropy', max_depth=10, min_samples_leaf=5,

                       min_samples_split=3, n_estimators=9)

clf_=clf.fit(X,y)
print(clf_.score(X,y))