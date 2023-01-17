# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv



from matplotlib import pyplot as plt

import matplotlib



matplotlib.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dd = pd.DataFrame({'A':[1,1,1,1,2,2,2,2,1,1,2,2], 'C': ['F','F','M','M','F','F','M','M','F','M','F','M'], 'B':[10,10,8,8,2,2,1,1,np.nan,np.nan,np.nan,np.nan]})

print(dd)

m = dd.groupby(['A','C'],as_index=False).median().sort_values(by='A', ascending=True)

print(m)

dd['B'].fillna(dd.groupby(['C','A'])['B'].transform('median'),inplace=True )

dd['Boy']=np.where((dd['B']>5) & (dd['B']<10),1,0)

print(dd)
train = pd.read_csv("../input/train.csv")

test = pd.read_csv('../input/test.csv')



print(train.shape)

print(test.shape)

print(list(train))
combined = train.copy()

combined = combined.append(test)

print(train.shape)

print(test.shape)

print(combined.shape)

print(test.iloc[0])

print(combined.iloc[891])
import re as re

def get_title(name):

	title_search = re.search(' ([A-Za-z]+)\.', name)

	# If the title exists, extract and return it.

	if title_search:

		return title_search.group(1)

	return ""





#train['Title'] = train['Name'].apply(get_title)

#test['Title'] = test['Name'].apply(get_title)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona":10}

    

full_data =[train,test]

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

#    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

# 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



#    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

#    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

#    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Title'] = dataset['Title'].map(title_mapping)

    #dataset['Title']=dataset['Title'].astype(int)

#print(train.head(5))

#print(test.head(5))

#print(pd.value_counts(test['Title']))

print(test[['Name','Title']].iloc[414])
train['Age'].fillna(train.groupby(['Sex','Pclass','Title'])['Age'].transform('median'),inplace=True )

test['Age'].fillna(train.groupby(['Sex','Pclass','Title'])['Age'].transform('median'),inplace=True )

#test['Age'].fillna(train['Age'].median(), inplace = True)

train['Fare'].fillna(train.groupby(['Pclass'])['Fare'].transform('median'), inplace = True)

test['Fare'].fillna(train.groupby(['Pclass'])['Fare'].transform('median'), inplace = True)

#print(train['Embarked'].describe())



#print(train.Embarked.dropna().mode()[0])

freqEmbark = train.Embarked.dropna().value_counts().index[0]

print(freqEmbark)

train['Embarked'].fillna(freqEmbark,inplace=True)

test['Embarked'].fillna(freqEmbark,inplace=True)

#print(test.Fare)

#print(train.isnull().sum())

#print(test.isnull().sum())

#engineer Age Bands

train['AgeBand']=pd.cut(train['Age'],5, labels=[0,1,2,3,4]).astype(int)

test['AgeBand']=pd.cut(test['Age'],5, labels=[0,1,2,3,4]).astype(int)

train['Fare'] = pd.qcut(train['Fare'],4,labels=[0,1,2,3]).astype(int)

test['Fare'] = pd.qcut(test['Fare'],4,labels=[0,1,2,3]).astype(int)

#print(test.Fare)

print(train[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Fare', ascending=True))

train['Sex'].replace({'male':1 ,'female':0},inplace=True)

test['Sex'].replace({'male':1 ,'female':0},inplace=True)

train['Embarked'].replace({'S':0,'C':1,'Q':2}, inplace = True)

test['Embarked'].replace({'S':0,'C':1,'Q':2}, inplace = True)
train['isAlone'] = np.where( (train['SibSp']+train['Parch'])==0,0,1)

test['isAlone'] = np.where( (test['SibSp']+test['Parch'])==0,0,1)

train['FamilySize']=train['Parch']+train['SibSp']+1

test['FamilySize']=test['Parch']+test['SibSp']+1

train['LargeFamily']=np.where(train['FamilySize']>5,1,0)

test['LargeFamily']=np.where(test['FamilySize']>5,1,0)

train['SmallFamily']=np.where((train['FamilySize']>=2) & (train['FamilySize']<=4),1,0)

test['SmallFamily']=np.where((test['FamilySize']>=2) & (test['FamilySize']<=4),1,0)





train.Age=train.Age.astype(int)

test.Age = test.Age.astype(int)



#print(test[['Family','SibSp','Parch']].head(10))

#train.drop(['Age','SibSp','Parch','Ticket'], inplace=True, axis=1)

#test.drop(['Age','SibSp','Parch','Ticket'], inplace=True, axis=1)

#train.Fare = train.Fare.astype(int)

#test.Fare = test.Fare.astype(int)

print(train.dtypes)

print(test.dtypes)

    

#test = pd.get_dummies(test,columns=['Title','Pclass','Embarked'])    

#train = pd.get_dummies(train,columns=['Title','Pclass','Embarked'])    

#print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

print(train.head(10))

print(test.head(10))

#print(pd.crosstab(train['Title'], train['Sex']))
from sklearn.model_selection import train_test_split



X = train.drop(['Survived','PassengerId','Name','Ticket','Age','SibSp','Parch','Cabin'], axis=1)

T = test.drop(['PassengerId','Name','Ticket','Age','SibSp','Parch','Cabin'], axis=1) ##'Age','SibSp','Parch',

Y = train['Survived'].values.ravel()

print(X.shape)

print(X)



print(T.isnull().sum())

#print(Y)

#print(T)



X_tr,X_val,Y_tr,Y_val =  train_test_split(X,Y,test_size=0.2, stratify =Y, random_state=123)
#feature selection

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel



clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')

clf.fit(X,Y)



features = pd.DataFrame()

features['importance']= clf.feature_importances_

features['features']= X.columns

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('features', inplace=True)

features.plot(kind='barh')
#model = SelectFromModel(clf, prefit=True)

#X_s = model.transform(X)

#print(X_s.shape)

#T_s = model.transform(T)
import xgboost as xg

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB



from sklearn.ensemble import VotingClassifier

from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import StratifiedKFold

from sklearn.metrics import auc

from sklearn.neighbors import KNeighborsClassifier

gbm = xg.XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.1, min_child_weight=7)

#gbm = xg.XGBClassifier()



gbm_params = {

    'learning_rate': [0.05, 0.1],

    'n_estimators': [300, 1000],

    'max_depth': [2, 3, 10],

    'min_child_weight': [1,3,5,7]

}



lr_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

# construct the set of hyperparameters to tune

knn_grid = {"n_neighbors": np.arange(1, 31, 2),

	"metric": ["euclidean", "cityblock"]}



clf1 = LogisticRegression(random_state=1, C=10)

clf2 = RandomForestClassifier(random_state=1,n_estimators=50, bootstrap=False, min_samples_split=10, min_samples_leaf=3,max_depth=6 )

clf3 = GaussianNB()

clf4 = KNeighborsClassifier(n_neighbors = 5, metric="cityblock")



eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),('knn', clf4) ,('xbboost', gbm)], voting='hard')





#cv = StratifiedKFold(Y)

#grid = GridSearchCV(clf4, knn_grid,cv=cv,verbose=10,n_jobs=-1)

#grid.fit(X_s, Y)

#print (grid.best_params_)







for clf, label in zip([clf1, clf2, clf3,clf4,gbm, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'KNN', 'XGBOOST','Ensemble']):

    scores = cross_val_score(clf, X, Y, scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))



gbm.fit(X,Y)

eclf.fit(X,Y)

predictions2=eclf.predict(T)

print("voting")

print(predictions2)

print('xb')

predictions= gbm.predict(T)

print(predictions)
submission1 = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions2

    })



submission1.to_csv('votinggood.csv', index=False)

submission2 = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions

    })

submission2.to_csv('xgbgood.csv', index=False)