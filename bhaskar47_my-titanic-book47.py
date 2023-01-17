# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

style.use('dark_background')

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir='/kaggle/input/titanic/'
train=pd.read_csv(data_dir+'train.csv')

test=pd.read_csv(data_dir+'test.csv')
train.describe()
test.describe()
test
otest=test.copy()

otrain=train.copy()
otrain
feature_nan=[feature for feature in train.columns if train[feature].isnull().sum()>=1]

for feature in feature_nan:

    print(f' {feature} : {np.round(np.round(train[feature].isnull().mean(),3)*100,2) } % missing')
train.columns
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(15,8))

women=train[train['Sex']=='female']

men=train[train['Sex']=='male']

ax=sns.distplot(women[women['Survived']==1].Age.dropna(),bins=18,label='Survived',ax=axes[0],kde=False)

ax=sns.distplot(women[women['Survived']==0].Age.dropna(),bins=18,label='Not_Survived',ax=axes[0],kde=False)

ax.legend()

ax.set_title('Female')

ax=sns.distplot(men[men['Survived']==1].Age.dropna(),bins=18,label='Survived',ax=axes[1],kde=False)

ax=sns.distplot(men[men['Survived']==0].Age.dropna(),bins=18,label='Not_Survived',ax=axes[1],kde=False)

ax.legend()

ax.set_title('Male')
train.columns
sns.FacetGrid(train,row="Embarked",size=7,aspect=1.6).map(sns.pointplot,'Pclass','Survived','Sex',palette=None,  order=None, hue_order=None ).add_legend()
sns.barplot(x='Pclass',y='Survived',data=train)

plt.legend()
sns.FacetGrid(train,row='Pclass',col='Survived',size=5,aspect=1.6).map(sns.distplot,'Age',bins=20,kde=False).add_legend()

train.columns
data=[train,test]

for dataset in data:

    dataset['relatives']=dataset['SibSp']+dataset['Parch']

    dataset.loc[dataset['relatives']>0,'not_alone']=1

    dataset.loc[dataset['relatives']==0,'not_alone']=0

    dataset['not_alone']=dataset['not_alone'].astype(int)

    

train['not_alone'] .value_counts()   
train.columns
sns.pointplot('relatives','Survived',data=train,size=100,aspect=150)
train=train.drop(['PassengerId'],axis=1)
train
train['Cabin'].unique()
import re

deck={"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"T":8,"U":0}

data=[train,test]

for dataset in data:

    dataset['Cabin']=dataset["Cabin"].fillna("U");

    dataset["Deck"]=dataset["Cabin"].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset["Deck"]=dataset["Deck"].map(deck)

    dataset["Deck"]=dataset["Deck"].astype(int)



train=train.drop(['Cabin'],axis=1)

test=test.drop(['Cabin'],axis=1)

    

    
train
data=[train,test]



for dataset in data:

    means=np.mean(dataset["Age"])

    std=np.std(dataset["Age"])

    nulls=dataset["Age"].isnull().sum()

    rand_age=np.random.randint(means-std,means+std,size=nulls)

    imputed_age=dataset["Age"].copy()

    #an_age=dataset["Age"].copy()

    imputed_age[np.isnan(imputed_age)]=rand_age

    dataset["Age"]=imputed_age

    dataset["Age"]=dataset["Age"].astype(int)



    
train["Embarked"].mode()
cv="S"

data=[train,test]

for dataset in data:

    dataset["Embarked"]=dataset["Embarked"].fillna(cv)
train.isnull().sum()
test.isnull().sum()
train.info()
data=[train,test]

for dataset in data:

    dataset["Fare"]=dataset["Fare"].fillna(0)

   
train["Name"]
titles={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Rare":5}

data=[train,test]

for dataset in data:

    dataset['Title']=dataset.Name.str.extract("([a-zA-Z]+)\.")

    dataset["Title"]=dataset["Title"].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')

    dataset['Title']=dataset['Title'].replace('Mile','Miss')

    dataset['Title']=dataset['Title'].replace('Ms','Miss')

    dataset['Title']=dataset['Title'].replace('Mme','Mrs')

    dataset['Title']=dataset['Title'].map(titles)

    dataset['Title']=dataset['Title'].fillna(0)

    

    

train=train.drop(['Name'],axis=1)

test=test.drop(['Name'],axis=1)
train['Title']=train['Title'].astype(int)
train
train.isnull().sum()
gender={"male":0,"female":1}

data=[train,test]

for dataset in data:

    dataset['Sex']=dataset['Sex'].map(gender)

    
train
train["Ticket"].describe()
train=train.drop(['Ticket'],axis=1)

test=test.drop(['Ticket'],axis=1)
train
ports={"S":0,"C":1,"Q":2}

data=[train,test]

for dataset in data:

    dataset["Embarked"]=dataset["Embarked"].map(ports)

    
train
train.isnull().sum()
train.info()
data=[train,test]

for dataset in data:

    dataset['Age']=dataset['Age'].astype(int)

    dataset.loc[dataset['Age']<=11,'Age']=0

    dataset.loc[(dataset['Age']>11) & (dataset['Age']<=18),'Age']=1

    dataset.loc[(dataset['Age']>18) & (dataset['Age']<=22),'Age']=2

    dataset.loc[(dataset['Age']>22) & (dataset['Age']<=27),'Age']=3

    dataset.loc[(dataset['Age']>27) & (dataset['Age']<=33),'Age']=4

    dataset.loc[(dataset['Age']>33) & (dataset['Age']<=40),'Age']=5

    dataset.loc[(dataset['Age']>40) & (dataset['Age']<=66),'Age']=6

    dataset.loc[ (dataset['Age']>66),'Age']=7

    



train['Age'].value_counts()
data=[train,test]

for dataset in data:

    #dataset['Age']=dataset['Age'].astype(int)

    dataset.loc[dataset['Fare']<=7.9,'Fare']=0

    dataset.loc[(dataset['Fare']>7.9) & (dataset['Fare']<=14.454),'Fare']=1

    dataset.loc[(dataset['Fare']>14.454) & (dataset['Fare']<=31),'Fare']=2

    dataset.loc[(dataset['Fare']>31) & (dataset['Fare']<=99),'Fare']=3

    dataset.loc[(dataset['Fare']>99) & (dataset['Fare']<=250),'Fare']=4

    dataset.loc[ (dataset['Fare']>250),'Fare']=5

    dataset['Fare']=dataset['Fare'].astype(int)
train
data=[train,test]

for dataset in data:

    dataset['Age_class']=dataset['Age']*dataset['Pclass']
train
data=[train,test]

for dataset in data:

    dataset['Fare_per_person']=dataset['Fare']/(dataset['relatives']+1)

    dataset['Fare_per_person']=dataset['Fare_per_person'].astype(int)
train
X=train.drop(['Survived'],axis=1)

y=train['Survived']

X
train['Survived'].value_counts()
processed_df=X
test=test.drop(['PassengerId'],axis=1)
test
from sklearn.preprocessing import StandardScaler

std_data=StandardScaler().fit_transform(X)

sample_data=std_data

print(np.round(np.mean(std_data),2))

print(np.std(std_data))
std_data_test=StandardScaler().fit_transform(test)

print(np.mean(std_data_test))

print(np.std(std_data_test))
from sklearn.manifold import TSNE

model=TSNE(n_components=2,perplexity=50,n_iter=5000,random_state=0)

tsne_data=model.fit_transform(std_data)

tsne_data=np.vstack((tsne_data.T,y)).T

tsne_df=pd.DataFrame(tsne_data,columns=['D1','D2','Class'])

print(f' Perplexity : {50} and iterations : {5000} ')

sns.FacetGrid(tsne_df,hue='Class',size=7).map(plt.scatter,'D1','D2').add_legend()

plt.show()
model=TSNE(n_components=2,perplexity=30,n_iter=5000,random_state=0)

tsne_data=model.fit_transform(std_data)

tsne_data=np.vstack((tsne_data.T,y)).T

tsne_df=pd.DataFrame(tsne_data,columns=['D1','D2','Class'])

print(f' Perplexity: {30} and iterations: {5000} ')

sns.FacetGrid(tsne_df,hue='Class',size=7).map(plt.scatter,'D1','D2').add_legend()

plt.show()
model=TSNE(n_components=2,perplexity=80,n_iter=5000,random_state=0)

tsne_data=model.fit_transform(std_data)

tsne_data=np.vstack((tsne_data.T,y)).T

tsne_df=pd.DataFrame(tsne_data,columns=['D1','D2','Class'])

print(f' Perplexity: {80} and iterations: {5000} ')

sns.FacetGrid(tsne_df,hue='Class',size=7).map(plt.scatter,'D1','D2').add_legend()

plt.show()
model=TSNE(n_components=2,perplexity=1000,n_iter=5000,random_state=0)

tsne_data=model.fit_transform(std_data)

tsne_data=np.vstack((tsne_data.T,y)).T

tsne_df=pd.DataFrame(tsne_data,columns=['D1','D2','Class'])

print(f' Perplexity: {100} and iterations: {5000} ')

sns.FacetGrid(tsne_df,hue='Class',size=7).map(plt.scatter,'D1','D2').add_legend()

plt.show()
from sklearn import decomposition

pca=decomposition.PCA()

pca.n_components=13

pca_data=pca.fit_transform(std_data)

percent_explained_variance=pca.explained_variance_/np.sum(pca.explained_variance_)

cummulative_explained_variance=np.cumsum(percent_explained_variance)

plt.plot(cummulative_explained_variance,linewidth=2)

plt.grid()

plt.xlabel('dimensions')

plt.ylabel('% explained variance')

plt.show()
pca=decomposition.PCA()

pca.n_components=8

final_data_train=pca.fit_transform(sample_data)

pca=decomposition.PCA()

pca.n_components=8

final_data_test=pca.fit_transform(std_data_test)
final_data_train.shape,final_data_test.shape
x_train=final_data_train

y_train=y

x_test=final_data_test
x_train.shape,y_train.shape,x_test.shape
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import accuracy_score


from sklearn.neighbors import KNeighborsClassifier

ml=list(range(100))

lst_k=[i for i in ml if i%2!=0 ]

cv_scores=[]

MSE=[]

for k in lst_k:

    knn=KNeighborsClassifier(n_neighbors=k)

    scores=cross_val_score(knn,x_train,y_train,cv=10,scoring='accuracy')

    cv_scores.append(scores.mean())

    

MSE=[1-x for x in cv_scores]

best_k=lst_k[MSE.index(min(MSE))]

print(f'best k of knn is {best_k}')

plt.plot(lst_k,MSE)

plt.xlabel('K value of KNN ->')

plt.ylabel('Mean Squared Error')

plt.show()

knn=KNeighborsClassifier(n_neighbors=best_k)

knn.fit(x_train,y_train)

y_pred_knn=knn.predict(x_test)

acc_knn=np.round(knn.score(x_train,y_train)*100,2)

print(f' The accuracy on train set of KNN model is {acc_knn}')
from sklearn.ensemble import RandomForestClassifier



# random_forest.fit(x_train,y_train)

# y_pred=random_forest.predict(x_test)

# acc_random_forest=np.round(random_forest.score(x_train,y_train)*100,2)

# print(f' The accuracy of random forest is {acc_random_forest}')





ml=list(range(150))

lst_n=[i for i in ml if i!=0]

cv_scores=[]

MSE=[]

for k in lst_n:

    #print(k,end='')

    random_forest=RandomForestClassifier(n_estimators=k)

    scores=cross_val_score(random_forest,x_train,y_train,cv=10,scoring='accuracy')

    cv_scores.append(scores.mean())

    

MSE=[1-x for x in cv_scores]

best_n=lst_n[MSE.index(min(MSE))]

print(f'best n of Random Forest is {best_n}')

plt.plot(lst_n,MSE)

plt.xlabel('n value of Random Forest ->')

plt.ylabel('Mean Squared Error')

plt.show()

random_forest=RandomForestClassifier(n_estimators=best_n)

random_forest.fit(x_train,y_train)

y_pred_rf=random_forest.predict(x_test)

acc_random_forest=np.round(random_forest.score(x_train,y_train)*100,2)

print(f' The accuracy of random forest is {acc_random_forest}')
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)

y_pred_dt=dt.predict(x_test)

acc_dt=np.round(dt.score(x_train,y_train)*100,2)

print(f' The accuracy of random forest is {acc_dt}')
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)

y_pred_nb=nb.predict(x_test)

acc_nb=np.round(nb.score(x_train,y_train)*100,2)

print(f' The accuracy of random forest is {acc_nb}')
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x_train, y_train)



y_pred_lr = logreg.predict(x_test)



acc_log = round(logreg.score(x_train, y_train) * 100, 2)

print(f'The accuracy of logistic Regression is {acc_log}')
from sklearn.svm import SVC, LinearSVC

linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)



Y_pred_svm = linear_svc.predict(x_test)



acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)

print(f'The accuracy of SVM is {acc_linear_svc}')
from sklearn.linear_model import SGDClassifier

sgd =SGDClassifier(max_iter=5, tol=None)

sgd.fit(x_train, y_train)

Y_pred_sgd= sgd.predict(x_test)



sgd.score(x_train, y_train)



acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)

print(f'The accuracy of SGD is {acc_sgd}')
results=pd.DataFrame({

    'Model':['SVM','KNN',"Logistic regression",'Random_forests',"Decision Tree",'SGD',"Naive Bayes"],

    'score':[acc_linear_svc,acc_knn,acc_log,acc_random_forest,acc_dt,acc_sgd,acc_nb]})

results=results.sort_values(by='score',ascending=False)

results=results.set_index('score')

results

    