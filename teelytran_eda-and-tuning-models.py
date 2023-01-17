import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")

df_target = pd.read_csv("../input/titanic/gender_submission.csv")
df_train.head()
df_train.columns
df_train.info()
df_train.dtypes.value_counts()
df_train.isnull().sum()
df_train.describe()
df_train.Sex.value_counts()
df_train.Embarked.value_counts()
df_train.Cabin.value_counts()
df_test.info()
df_test.describe()
df_train[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False)
df_train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False)
df_train[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)
df_train[['Parch', 'Survived']].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False)
df_train.Ticket.nunique
pd.DataFrame.hist(df_train, figsize=[15,15])

plt.show()
plt.figure(figsize=(30,30))

sns.set(rc={'figure.figsize':(20,20)})
sns.catplot(x='Sex',y='Survived',data=df_train,kind='bar')
sns.catplot(x='Pclass',y='Survived',data=df_train,kind='bar')
sns.set(rc={'figure.figsize':(10,10)})

sns.countplot(df_train.Embarked)
sns.catplot(x = 'Embarked',y='Survived',kind = 'point', data = df_train, hue = 'Sex')
sns.catplot(x='Sex',y='Survived', hue='Pclass', col= 'Pclass',data=df_train,kind='bar',color='g')
sns.catplot(x='Survived',y='Age', data=df_train)
grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', height=4)

grid.map(plt.hist,'Age', bins=20)

grid.add_legend();
sns.catplot(x='Survived',y='Fare', data=df_train)
sns.catplot(x='Survived', y='SibSp', data=df_train)
sns.catplot(x="Survived", y="Parch", data=df_train)
df = pd.concat([df_train, df_test],axis=0)

df.drop(['Survived'], axis=1, inplace=True)

df = df.drop(['PassengerId','Cabin','Ticket'], axis=1)
df.Age.fillna(df.Age.median(), inplace=True)
df.Fare.fillna(df.Fare.median(), inplace=True)
df.Embarked.fillna(df.Embarked.value_counts().index[0], inplace=True)

df=pd.get_dummies(df, columns=['Embarked'])
name_split=df.Name.str.split(",", n=1,expand=True)

name_split.head()

df["LName"]=name_split[0]

name_split2 = name_split[1].str.split(".", n=1, expand =True)

name_split2.head()

df['Title']=name_split2[0]

df['FName']=name_split2[1]
df.Title.value_counts()
df.replace([' Don', ' Rev', ' Dr', ' Mme',

        ' Major', ' Sir', ' Col', ' Capt',' Jonkheer'],'Others(M)', inplace=True)

df.replace([' Ms', ' Lady', ' Mlle',' the Countess', ' Dona'], 'Others(F)', inplace=True)

df.drop(['LName','FName','Name'],axis=1, inplace=True)
df = pd.get_dummies(df, columns=['Title'])
df['Family']=df['SibSp']+df['Parch']+1
df['Single']=df.Family.map(lambda s:1 if s==1 else 0)

df['SmallF']=df.Family.map(lambda s:1 if s==2 else 0)

df['MedF']=df.Family.map(lambda s:1 if 3<=s<=4 else 0)

df['LargeF']=df.Family.map(lambda s:1 if s>=5 else 0)
df.drop(['Family'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['Sex'], prefix="Gender")
df.head()
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn import metrics

from scipy.stats import randint

X_train=df[:891]

X_test=df[891:]

y_train=df_train.Survived

X_train.shape, X_test.shape, y_train.shape
y_test=df_target.Survived.values.reshape(-1,1)
stanScaler= StandardScaler()

X_train_scaled=stanScaler.fit_transform(X_train)

X_test_scaled=stanScaler.fit_transform(X_test)
Accuracy=[]
param_grid ={'n_neighbors':np.arange(1,50)}

knn=KNeighborsClassifier()

knn_cv=GridSearchCV(knn, param_grid, cv=10)

knn_cv.fit(X_train_scaled, y_train)

best_KNN=knn_cv.best_estimator_

y_pred = best_KNN.predict(X_test_scaled)

accu_score= accuracy_score(y_test,y_pred) 

Accuracy.append(accu_score)

print("Accuracy Score is {}".format(accu_score))

print("KNN best parameter is {}".format(knn_cv.best_params_))
submission1=pd.DataFrame(columns=['PassengerId','Survived'])

submission1.PassengerId=df_test.PassengerId

submission1.Survived=y_pred

submission1.to_csv("submission1.csv", index=False)
y_pred_prob = best_KNN.predict_proba(X_test_scaled)[:,1]

fpr, tpr, thresholds= roc_curve(y_test, y_pred_prob)

plt.plot([0,1],[1,0],'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title("ROC Curve")

plt.show()
print("The area under the curve is {}".format(roc_auc_score(y_test, y_pred_prob)))
c_space = np.logspace(-5, 8, 15)

param_grid = {'C': c_space}

logreg = LogisticRegression(max_iter=1000)

logreg_cv = GridSearchCV(logreg, param_grid, cv=10)

logreg_cv.fit(X_train_scaled,y_train)

best_LR = logreg_cv.best_estimator_

y_pred = best_LR.predict(X_test_scaled)

accu_score = accuracy_score(y_test, y_pred)

Accuracy.append(accu_score)

print("Accuracy score is {}".format(accu_score))

print("Logistic Regression best parameter is {}".format(logreg_cv.best_params_))
submission2=pd.DataFrame(columns=['PassengerId','Survived'])

submission2.PassengerId=df_test.PassengerId

submission2.Survived=y_pred

submission2.to_csv("submission2.csv", index=False)
param_dist = {"max_depth": [3, 4, 5],

              "max_features": randint(1, 9),

              "min_samples_leaf": randint(1, 9),

              "criterion": ["gini", "entropy"]}

dt = DecisionTreeClassifier()

dt_cv=RandomizedSearchCV(dt,param_dist,cv=10)

dt_cv.fit(X_train, y_train)

best_dt=dt_cv.best_estimator_

best_dt.predict(X_test)

accu_score=accuracy_score(y_test, y_pred)

Accuracy.append(accu_score)

print("Accuracy score is {}".format(accu_score))

print("Decision Tree best parameter is {}".format(dt_cv.best_params_))

submission3=pd.DataFrame(columns=['PassengerId','Survived'])

submission3.PassengerId=df_test.PassengerId

submission3.Survived=y_pred

submission3.to_csv("submission3.csv", index=False)
param_grid={"max_depth":[3,4,5], 

            "n_estimators":[100,350,500],

            "min_samples_leaf":[2,10,30],

            "max_features":["sqrt","log2"]

           }

gb= GradientBoostingClassifier()

cv_gb=GridSearchCV(gb, param_grid,cv=10)

cv_gb.fit(X_train,y_train)

best_gb=cv_gb.best_estimator_

y_pred=best_gb.predict(X_test)

accu_score=accuracy_score(y_test, y_pred)

Accuracy.append(accu_score)

print("Accuracy score is {}".format(accu_score))

print("Gradient Boosting best parameter is {}".format(cv_gb.best_params_))

submission4=pd.DataFrame(columns=['PassengerId','Survived'])

submission4.PassengerId=df_test.PassengerId

submission4.Survived=y_pred

submission4.to_csv("submission4.csv", index=False)
models = ['Logistic Regression(StdScaler)','K-Nearest Neighbors(StdScaler)','Decision Tree','GradientBoosting']

total2=list(zip(models,Accuracy))

result2 = pd.DataFrame(total2, columns=['Models after HT','Accuracy'])

result2.sort_values(by='Accuracy', ascending=False)