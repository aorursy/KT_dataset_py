# import packages
import pandas as pd
import numpy as np
import re #regex for future title extraction from Name
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
#import data set
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.info()
train.head()
train = train.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)
test = test.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)

train=pd.get_dummies(train,columns=['Sex','Embarked'],drop_first=True)
test=pd.get_dummies(test,columns=['Sex','Embarked'],drop_first=True)
#MICE package from fancyimpute
from fancyimpute import MICE
#We use the train dataframe from Titanic dataset
#fancy impute removes column names.
train_cols = list(train)

#Use MICE to fill in each row's missing features
train = pd.DataFrame(MICE(verbose=False).complete(train))
train.columns = train_cols

test_cols = list(test)
test=pd.DataFrame(MICE(verbose=False).complete(test))
test.columns = test_cols
train['AgeBand'] = pd.cut(train['Age'], 5)
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
titanic = [train,test]
for df in titanic:    
    df.loc[ df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age']

train = train.drop(['AgeBand'],axis=1)
train.head()
fare = train.loc[:,'Fare'].values
fare = preprocessing.scale(fare)
train.loc[:,'Fare'] = fare


fare = test.loc[:,'Fare'].values
fare = preprocessing.scale(fare)
test.loc[:,'Fare'] = fare
train.info()
sns.heatmap(train.corr(), annot=True, fmt=".2f")
plt.show()
pca = PCA(n_components=8)
pca.fit(train)
#The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

plt.plot(var1)
pca = PCA(n_components=5)
Y = train[['Survived']]
train1 = train.drop('Survived', axis = 1)
X = pca.fit_transform(train1)
test1 = pca.fit_transform(test)

logreg = LogisticRegression()
logreg.fit(X, Y)
Ypred = logreg.predict(test1)
acc_log = logreg.score(X,Y)
print(acc_log)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X,Y)
Ypred1 = decision_tree.predict(test1).astype(int)
acc_decision_tree = decision_tree.score(X,Y) 
print(acc_decision_tree)
test2 = pd.read_csv('../input/test.csv')
test = test2.loc[:,['PassengerId']]
test['Survived']=Ypred1[:]
test.to_csv('joel.csv',index=False)