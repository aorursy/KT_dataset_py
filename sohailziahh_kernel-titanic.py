import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
import cufflinks as cf

cf.go_offline()
train['Fare'].iplot(kind='hist',bins=30,color='green')
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)
train.head()
train.dropna(inplace=True)
train.info()
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), train['Survived'], test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
test = pd.read_csv('../input/test.csv')
test.drop('Cabin',axis=1,inplace=True)
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
test['Fare'].fillna((test['Fare'].mean()), inplace=True)
sex = pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test= pd.concat([test,sex,embark],axis=1)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
test_pred = logmodel.predict(test)
from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(n_estimators=300)
rfmodel.fit(X_train, y_train)
pred_rf = rfmodel.predict(X_test)
print(classification_report(pred_rf, y_test))
rfmodel.score(X_test, y_test)
pred_results = rfmodel.predict(test)
test['Survived'] = pred_results
submission = test[['PassengerId', 'Survived']]
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print(classification_report(y_test, pred))

error_rate = []



for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=[10,6])

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')


knn = KNeighborsClassifier(n_neighbors=25)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)
print(classification_report(y_test, pred))

#catboost

train = pd.read_csv('../input/train.csv')

test =  pd.read_csv('../input/test.csv')
train.fillna(-999,inplace=True)

test.fillna(-999,inplace=True)
x = train.drop('Survived',axis=1)

y = train.Survived
x.dtypes
cat_features_index = np.where(x.dtypes != float)[0]
import hyperopt

from catboost import Pool, CatBoostClassifier, cv
model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=101)
model.fit(X_train,y_train,cat_features=cat_features_index,eval_set=(X_test,y_test))
from sklearn.metrics import accuracy_score

print('the test accuracy is :{:.6f}'.format(accuracy_score(y_test,model.predict(X_test))))
pred = model.predict(test)

pred = pred.astype(np.int)

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred})
submission.to_csv('catboost.csv',index=False)
