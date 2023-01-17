import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt



from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print(train.shape)

print(test.shape)
test.drop(['Ticket','Cabin'],axis=1, inplace=True)

test.set_index('PassengerId',inplace=True)



train.drop(['Ticket','Cabin'],axis=1, inplace=True)

train.set_index('PassengerId',inplace=True)



train.head()
test['title'] = test.Name.str.extract('([A-Za-z]+)\.')

test['title'].replace(['Mlle','Mme','Ms','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona','Dr'],['Miss','Miss','Miss','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Miss','Mr'],inplace=True)

test = pd.concat([test.drop(['title','Name'],axis=1), pd.get_dummies(test[['title']], drop_first=True)], axis=1)



train['title'] = train.Name.str.extract('([A-Za-z]+)\.')

train['title'].replace(['Mlle','Mme','Ms','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona','Dr'],['Miss','Miss','Miss','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Miss','Mr'],inplace=True)

train = pd.concat([train.drop(['title','Name'],axis=1), pd.get_dummies(train[['title']], drop_first=True)], axis=1)



train.head()
features_numeric = ['Age','SibSp','Parch','Fare']

features_label  = ['Pclass','Sex','Embarked']



test = test.fillna(test[features_numeric].mean())

test = test.fillna(test[features_label].mode().iloc[0])



train = train.fillna(train[features_numeric].mean())

train = train.fillna(train[features_label].mode().iloc[0])



train.info()
test = pd.concat([test.drop(['Sex'],axis=1), pd.get_dummies(test[['Sex']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['Sex'],axis=1), pd.get_dummies(train[['Sex']], drop_first=True)], axis=1)



test = pd.concat([test.drop(['Embarked'],axis=1), pd.get_dummies(test[['Embarked']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['Embarked'],axis=1), pd.get_dummies(train[['Embarked']], drop_first=True)], axis=1)



test['Family'] = test['SibSp'] + test['Parch']

test['Alone'] = 0

test.loc[test.Family==0,'Alone'] = 1

test.drop(['SibSp','Parch'],axis=1,inplace=True)



train['Family'] = train['SibSp'] + train['Parch']

train['Alone'] = 0

train.loc[train.Family==0,'Alone'] = 1

train.drop(['SibSp','Parch'],axis=1,inplace=True)



train.head()
#test['Fare_scaled'] = 0

#test['Age_scaled'] = 0

#test[['Fare_scaled','Age_scaled']] = StandardScaler().fit_transform(test[['Fare','Age']])

#test.drop(['Fare','Age'],axis=1,inplace=True)



#train['Fare_scaled'] = 0

#train['Age_scaled'] = 0

#train[['Fare_scaled','Age_scaled']] = StandardScaler().fit_transform(train[['Fare','Age']])

#train.drop(['Fare','Age'],axis=1,inplace=True)



#train.head()
print(train.shape)

print(test.shape)
plt.figure(figsize=(8,6))

sns.heatmap(train.corr(),cmap='summer_r');
X,y = train.drop(['Survived'],axis=1), train['Survived'] 



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



print(X.shape)

print(X_train.shape)

print(X_test.shape)

print(y.shape)

print(y_train.shape)

print(y_test.shape)
classifiers = [

    KNeighborsClassifier(3), SVC(probability=True), DecisionTreeClassifier(), RandomForestClassifier(), AdaBoostClassifier(),

    GradientBoostingClassifier(), GaussianNB(), LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(), LogisticRegression(),

    XGBClassifier()

]



log_cols = ["Classifier", "Accuracy"]

log = pd.DataFrame(columns=log_cols)

acc_dict = {}



for clf in classifiers:

    name = clf.__class__.__name__

    clf.fit(X_train, y_train)

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    if name in acc_dict:

        acc_dict[name] += acc

    else:

        acc_dict[name] = acc



for clf in acc_dict:

    acc_dict[clf] = acc_dict[clf]

    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

    log = log.append(log_entry)





plt.figure(figsize=(10,10))

plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')

plt.grid(True)

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b");
xgb = XGBClassifier()#max_depth = 5, n_estimators=1000, learning_rate=0.2, nthread=3, subsample=1.0, colsample_bytree=0.5, min_child_weight=3, reg_alpha=0.03)

xgb.fit(X_train, y_train, early_stopping_rounds=50, eval_metric='auc', eval_set=[(X_train, y_train),(X_test, y_test)])
survivors = xgb.predict(test)

submission = pd.DataFrame({'PassengerId':test.index,'Survived':survivors})

submission.head()
#gbc = GradientBoostingClassifier(learning_rate=0.11).fit(X_train,y_train)

#gbc.score(X_test,y_test)
#params = {

#    "learning_rate": [0.01, 0.05, 0.1, 0.25, 0.5, 1],

#    "n_estimators": [90,100,110],

#    "max_depth": [3,4,5],

#    "min_samples_split": np.linspace(0.1, 1.0, 10, endpoint=True),

#    "min_samples_leaf": np.linspace(0.1, 0.5, 5, endpoint=True),

#    "max_features": list(range(1,train.shape[1]))

#}

#gbc = GradientBoostingClassifier()

#grid = GridSearchCV(gbc, cv=5, param_grid=params)

#grid.fit(X,y)



#print(grid.best_score_)

#print(grid.best_params_)
#gbc = GradientBoostingClassifier(learning_rate=0.25,max_depth=5,max_features=5,min_samples_leaf=0.1,min_samples_split=0.1,n_estimators=90).fit(X,y)

#survivors = gbc.predict(test)

#submission = pd.DataFrame({'PassengerId':test.index,'Survived':survivors})

#submission.head()
submission.to_csv('Titanic_Survivors_Predictions.csv',index=False)