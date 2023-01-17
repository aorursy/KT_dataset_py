# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn import metrics
%matplotlib inline
from scipy import stats
from statsmodels.stats.weightstats import ztest
import plotly.graph_objs as go
import plotly.express as px 
from itertools import cycle
dftrain=pd.read_csv("/kaggle/input/titanic/train.csv")
dftrain
dftest=pd.read_csv("/kaggle/input/titanic/test.csv")
dftest
dftest.loc[dftest['Age'] < 18, "Age_category"]="Child"
dftest.loc[dftest['Age'] >=18, "Age_category"]="Adult"
dftest.head()
sns.heatmap(dftrain.isnull(), cbar=False)
print("Nan values in PassengerId:",dftrain["PassengerId"].isnull().sum())
print("Nan values in Survived:", dftrain["Survived"].isnull().sum())
print ("Nan values in Pclass:", dftrain["Pclass"].isnull().sum())
print("Nan values in Name:",dftrain["Name"].isnull().sum())
print("Nan values in Sex:", dftrain["Sex"].isnull().sum())
print ("Nan values in Age:", dftrain["Age"].isnull().sum())
print("Nan values in SibSp:",dftrain["SibSp"].isnull().sum())
print("Nan values in Parch:", dftrain["Parch"].isnull().sum())
print ("Nan values in Ticket:", dftrain["Ticket"].isnull().sum())
print("Nan values in Fare:",dftrain["Fare"].isnull().sum())
print("Nan values in Cabin:", dftrain["Cabin"].isnull().sum())
print ("Nan values in Embarked:", dftrain["Embarked"].isnull().sum())
mean=dftrain["Age"].mean()
dftrain["Age"].replace(np.nan,mean,inplace=True)
print("Nan values in Age:",dftrain["Age"].isnull().sum())
em=pd.isnull(dftrain["Embarked"])
dftrain[em]
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=dftrain, palette="Set1")
dftrain["Embarked"].replace(np.nan,'C',inplace=True)
print("Nan values in Embarked:",dftrain["Embarked"].isnull().sum())
dftrain.loc[dftrain['Age'] < 18, "Age_category"]="Child"
dftrain.loc[dftrain['Age'] >=18, "Age_category"]="Adult"
dftrain
dftrain["Age_category"].value_counts()
A=dftrain["Sex"].value_counts(normalize=True).rename_axis('Sex').reset_index(name='count')
A
A.plot.bar(x="Sex", y="count", rot=5, title="Male and female on the ship")

plt.show(block=True)
sns.kdeplot(dftrain['Age'])
dftrain["Age_category"].value_counts(normalize=True)
sns.countplot(x= 'Age_category', hue= 'Sex', data=dftrain)
B=dftrain["Pclass"].value_counts(normalize=True).rename_axis('Pclass').reset_index(name='count')
B
B.plot.bar(x="Pclass", y="count", rot=10, title="class wise passenger")

plt.show(block=True)
C = dftrain[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
C
dftrain["Fare"].describe()
f=dftrain.loc[dftrain["Fare"]==0]
f["Age_category"].value_counts(normalize=True)
f["Pclass"].value_counts(normalize=True)
f["Embarked"].value_counts(normalize=True)
f["Sex"].value_counts(normalize=True)
f["Survived"].value_counts()
sns.kdeplot(dftrain['Fare'])
dftrain["Survived"]= dftrain["Survived"].apply(str)     #changing Survived into str 
dftrain["Survived"] =dftrain["Survived"].replace(["1","0"],["survived", "dead"])
dftrain.head()
D= dftrain["Survived"].value_counts(normalize=True).rename_axis('Survived').reset_index(name='count')
D
D.plot.bar(x="Survived", y="count", rot=10, title="Survival")

plt.show(block=True)
C.plot.bar(x="Sex", y="Survived", rot=10, title="Gender wise survival")

plt.show(block=True)
fig = px.histogram(dftrain, x='Age', color='Survived')
fig.show()
S=dftrain.loc[dftrain["Survived"]=="survived"]
S["Age_category"].value_counts(normalize=True)
age1=dftrain["Age_category"].value_counts().rename_axis('Age_category').reset_index(name='count')
age2=S["Age_category"].value_counts().rename_axis('Age_category').reset_index(name='countSV')
AC_survival= pd.merge(age1, age2)
AC_survival
AC_survival['Percent'] = (AC_survival['countSV'] / 
                  AC_survival['count']) * 100
AC_survival
fig = px.histogram(dftrain, x='Age_category', color='Survived')
fig.show()
dftrain.groupby(['Sex'])['Survived'].value_counts(normalize=True)
sns.countplot(x='Survived', hue='Sex', data=dftrain)
bins = np.linspace(dftrain.Age.min(), dftrain.Age.max(), 5)
g = sns.FacetGrid(dftrain, col="Sex", hue="Survived", palette="Set1",col_wrap=2) 
g.map(plt.hist, 'Age', bins=bins,ec="k")

g.axes[-1].legend()
plt.show()
plt.scatter(dftrain["Age"],dftrain["Fare"])
plt.show
dftrain[dftrain["Fare"]>=500]
fig = px.histogram(dftrain, x='Embarked', color='Pclass')
fig.show()
bins = np.linspace(dftrain.Pclass.min(), dftrain.Pclass.max(), 8)
g = sns.FacetGrid(dftrain, col="Sex", hue="Survived", palette="Set1",col_wrap=2) 
g.map(plt.hist, 'Pclass', bins=bins,ec="k")

g.axes[-1].legend()
plt.show()
E=dftrain["Pclass"].value_counts().rename_axis('Pclass').reset_index(name='count').sort_values(by="Pclass")
F=S["Pclass"].value_counts().rename_axis('Pclass').reset_index(name='countSV').sort_values(by="Pclass")
Class_survival= pd.merge(E, F)
Class_survival
Class_survival['Percent'] = (Class_survival['countSV'] / 
                  Class_survival['countSV'].sum()) * 100
Class_survival
Class_survival.plot.bar(x="Pclass", y="Percent", rot=10, title="class wise survival")

plt.show(block=True)
dftrain.groupby(['Pclass'])['Survived'].value_counts(normalize=True)
sns.countplot(x='Survived', hue='Pclass', data=dftrain)
dftrain["Survived"] =dftrain["Survived"].replace(["survived","dead"],["1", "0"])

dftrain["Sex"] =dftrain["Sex"].replace(["male","female"],["1", "0"])
X = dftrain[["PassengerId", "Pclass", "Sex", "Age", "Fare"]]
X[0:5]
y = dftrain['Survived'].values
y[0:5]
from sklearn import preprocessing as pr
X= pr.StandardScaler().fit(X).transform(X)
X[0:5]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.15, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat = neigh.predict(X_test)
yhat[0:5]
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
dftest["Sex"] =dftest["Sex"].replace(["male","female"],["1", "0"])
Feature1=dftest[['PassengerId', "Pclass", "Sex", "Age", "Fare"]]
Feature1.dropna(subset=["Age","Fare"], inplace=True)
print("nan values in fare", Feature1["Fare"].isnull().sum())
Feature2= pr.StandardScaler().fit(Feature1).transform(Feature1)
Feature2[0:5]
X_test1=Feature2
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks): 
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
mean_acc
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
from sklearn.tree import DecisionTreeClassifier
survivalTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
survivalTree.fit(X_train,y_train)
survivalTree 
predTree = survivalTree.predict(X_test)
print (predTree [0:10])
print (y_test [0:10])
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))
from sklearn import svm
SVM_classifier = svm.SVC()
SVM_classifier.fit(X_train, y_train) 
svmyhat = SVM_classifier.predict(X_test)
print(y_test[0:10])
print(svmyhat[0:10])
print("SVM Accuracy: ", metrics.accuracy_score(y_test, svmyhat))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR
testyhat = LR.predict(X_test)
testyhat[0:5]
print("LR Accuracy: ", metrics.accuracy_score(y_test, testyhat))
yhat_prob = LR.predict_proba(X_test)
yhat_prob[0:5]
rep = pd.DataFrame([], columns = ['Algorithm' , "Accuracy"]) 

rep   = rep.append(pd.Series(['KNN', metrics.accuracy_score(y_test, yhat)], index=rep.columns ), ignore_index=True)
rep   = rep.append(pd.Series(['Decision Tree', metrics.accuracy_score(y_test, predTree)], index=rep.columns ), ignore_index=True)
rep   = rep.append(pd.Series(['SVM', metrics.accuracy_score(y_test, svmyhat)], index=rep.columns ), ignore_index=True)
rep   = rep.append(pd.Series(['LogisticRegression', metrics.accuracy_score(y_test, testyhat)], index=rep.columns ), ignore_index=True)

rep.round(2)
treeyhat = survivalTree.predict(X_test1)
treeyhat [0:5]
k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat1 = neigh.predict(X_test1)
yhat1[0:5]
p1=pd.DataFrame( data= Feature1["PassengerId"], columns=["PassengerId"])
p2=pd.DataFrame(data= yhat1, columns=["Survived"])
#import sqlalchemy as sa
#engine = sa.create_engine('sqlite:///tmp.db')
#p1.to_sql('p1', engine)
#p2.to_sql('p2', engine)
#predict = pd.read_sql_query('SELECT * FROM p1 JOIN p2', engine)
#predict

#p1['tmp'] = 1
#p2['tmp'] = 1
#predict = pd.merge(p1, p2, on=['tmp'])
#predict.drop_duplicates(subset="PassengerId", inplace=True)
#predict = predict.drop('tmp', axis=1)
#predict
predict["Survived"].value_counts(normalize=True)
