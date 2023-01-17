#importing the libraries that we use

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#reading train data and test data

data=pd.read_csv('/kaggle/input/titanic/train.csv')

datatest=pd.read_csv('/kaggle/input/titanic/test.csv')

data.head()
data.info()
data.columns
data.shape
numerical_cols = [col for col in data.columns if data[col].dtype != 'object']

categorical_cols = [col for col in data.columns if data[col].dtype == 'object']

numerical_cols, categorical_cols
data.isnull().sum()
sns.pairplot(data)
data.info()
features = ["Pclass", "SibSp", "Parch"]

data1 = data

pd.get_dummies(data1[features])

sns.heatmap(data1.corr(),annot=True, fmt = ".2f", cmap = "hot")
sns.distplot(data['Age'])
sns.distplot(data['Fare'])
sns.factorplot("Pclass", "Survived",   data=data, kind="bar", palette="muted", legend=True)
sns.factorplot(y="Age",x="Sex",hue="Pclass", data=data,kind="box")
sns.factorplot("Embarked", "Survived", data=data, kind="bar", palette="muted")
sns.factorplot("Parch", "Survived",   data=data, kind="bar", palette="muted",)
sns.factorplot("SibSp", "Survived",   data=data, kind="bar", palette="muted", legend=True)
sns.factorplot("Sex", "Survived",  data=data, kind="bar", palette="muted")
sns.factorplot(y="Age",x="Sex",data=data,kind="box")
#Cabin 

data['Cabin Section']= data['Cabin'].str[0]

data['Cabin Section']



#Title

data['Title3']= data['Name'].apply(lambda x: x.strip())

data['Title3']

data['Title2']= data['Title3'].apply(lambda x: x.split(',')[1])

data['Title2']

data['Title']= data['Title2'].apply(lambda x: x.split('.')[0])

data['Title']



data.drop(columns=['Title3','Title2','Cabin','Name'],inplace=True)
sns.countplot(data['Cabin Section'],hue=data['Survived'])
sns.factorplot("Cabin Section", "Survived", data=data, kind="bar",hue='Pclass' , palette="muted", legend=True)
chart=sns.countplot(data['Title'])

chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
Other = [' Rev', ' Dr',' Mme', ' Ms', ' Major', ' Lady', ' Sir', ' Mlle', ' Col', ' Capt',

 ' the Countess', ' Jonkheer']



for i in Other:

    data['Title'] = data['Title'].apply(lambda x:x.replace(i, 'Other'))
chart=sns.countplot(data['Title'])

chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
sns.factorplot("Title", "Survived", data=data, kind="bar",hue='Pclass' , palette="muted", legend=True)
def remove_outlier(data, col_name):

    q1 = data[col_name].quantile(0.25)

    q3 = data[col_name].quantile(0.75)

    iqr = q3-q1 #Interquartile range

    lwb  = q1-1.5*iqr

    upb = q3+1.5*iqr

    df_out = data.loc[(data[col_name] >lwb) | (data[col_name] < upb)]

    return df_out



data2=remove_outlier(data,'Age')

data2



data3=remove_outlier(data2,'SibSp')

data3



data4=remove_outlier(data3,'Parch')

data4



data5=remove_outlier(data3,'Fare')

data5

data5['Age'].fillna(data5['Age'].mean,inplace=True)

data5['Embarked']=data5['Embarked'].fillna(data5['Embarked'].mode,inplace=True)
data5['Fare']=data5['Fare'].apply(np.log)

data5['Fare']=data5['Fare'].apply(np.log).hist()
from sklearn.model_selection import KFold, cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression



kfold=KFold(n_splits=5)



y = data5["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(data5[features])

X_test = pd.get_dummies(datatest[features])

random_state = 42

classifier = [RandomForestClassifier(random_state = random_state),

             LogisticRegression(random_state = random_state),

             KNeighborsClassifier()]





for i in range(len(classifier)):

    classifier[i].fit(X,y)

    print(cross_val_score(classifier[i],X,y,cv=kfold))

    print(np.mean(cross_val_score(classifier[i],X,y,cv=kfold)))   
from sklearn.model_selection import GridSearchCV

rfc=RandomForestClassifier(random_state=42)

param_grid = { 

    'n_estimators': [5, 10],

    'max_depth':[1,10],

    'max_features':[2,3],

    'max_samples':[200,400]

}



CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

g=CV_rfc.fit(X, y)

print('n_estimators:', g.best_estimator_.get_params()['n_estimators'])

print('max_depth:', g.best_estimator_.get_params()['max_depth'])

print('max_features:', g.best_estimator_.get_params()['max_features'])

print('max_samples:', g.best_estimator_.get_params()['max_samples'])

print('train_score:', CV_rfc.score(X, y))
knn=KNeighborsClassifier()



param_grid = { 

    'n_neighbors':[1,50],

    'leaf_size':[1,30],

    'p':[1,2]

}

CV_knn = GridSearchCV(estimator=knn, param_grid=param_grid, cv= 5)

g=CV_knn.fit(X, y)

print('Best leaf_size:', g.best_estimator_.get_params()['leaf_size'])

print('Best p:', g.best_estimator_.get_params()['p'])

print('Best n_neighbors:', g.best_estimator_.get_params()['n_neighbors'])

print('train_score:', CV_knn.score(X, y))
lr=LogisticRegression()



param_grid = { 

    'C':[1,30],

}

CV_lr = GridSearchCV(estimator=lr, param_grid=param_grid, cv= 5)

g=CV_lr.fit(X, y)

print('C:', g.best_estimator_.get_params()['C'])

print('train_score:', CV_lr.score(X, y))
from sklearn.ensemble import VotingClassifier



CV_rfc.fit(X,y)

CV_knn.fit(X,y)

CV_lr.fit(X,y)



pred1=CV_rfc.predict(X_test)

pred2=CV_knn.predict(X_test)

pred3=CV_lr.predict(X_test)



models = [('lr',CV_lr),('knn',CV_knn)]

ensemble = VotingClassifier(estimators=models, voting='hard')

ensemble.fit(X,y)

ensemble.predict(X_test)

print('train_score:', ensemble.score(X,y))
pred1=CV_rfc.predict(X_test)

pred2=CV_knn.predict(X_test)

pred3=CV_lr.predict(X_test)



models = [('lr',CV_lr),('rfc',CV_rfc)]

ensemble = VotingClassifier(estimators=models, voting='hard')

ensemble.fit(X,y)

ensemble.predict(X_test)

print('train_score:', ensemble.score(X,y))
pred1=CV_rfc.predict(X_test)

pred2=CV_knn.predict(X_test)

pred3=CV_lr.predict(X_test)



models = [('knn',CV_knn),('rfc',CV_rfc)]

ensemble = VotingClassifier(estimators=models, voting='hard')

ensemble.fit(X,y)

ensemble.predict(X_test)

print('train_score:', ensemble.score(X,y))
pred1=CV_rfc.predict(X_test)

pred2=CV_knn.predict(X_test)

pred3=CV_lr.predict(X_test)



models = [('knn',CV_knn),('rfc',CV_rfc),('lr',CV_lr)]

ensemble = VotingClassifier(estimators=models, voting='hard')

ensemble.fit(X,y)

ensemble.predict(X_test)

print('train_score:', ensemble.score(X,y))
CV_rfc.fit(X,y)

CV_knn.fit(X,y)

CV_lr.fit(X,y)



pred1=CV_rfc.predict(X_test)

pred2=CV_knn.predict(X_test)

pred3=CV_lr.predict(X_test)



final_pred=(pred1+pred2+pred3)/3

print('train_score:', (CV_rfc.score(X,y)+CV_knn.score(X,y)+CV_lr.score(X,y))/3)
## 7. Submission
output = pd.DataFrame({'PassengerId': datatest.PassengerId, 'Survived': CV_rfc.predict(X_test)})

output.to_csv('gender_submission.csv', index=False)