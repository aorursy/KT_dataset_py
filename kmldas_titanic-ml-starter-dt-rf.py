import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns





from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier 

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV



from sklearn import tree

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.isnull().sum() # To check No missing values 
sns.boxplot(x='Pclass', y='Age', data = train)

plt.show()
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
train.drop('Cabin',axis=1,inplace=True) 

train
train.dropna(inplace = True)

train
train.info()


sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)



train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train
train = pd.concat([train,sex,embark],axis=1)



train_data = train.drop('Survived', axis=1)

label = train['Survived']
train_data
train_data.info()
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
DT_clf = tree.DecisionTreeClassifier(random_state=0)

clf = DT_clf.fit(train_data, label)

tree.plot_tree(clf)

fn=list(train_data.columns)

cn=["0","1"]

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (2,2), dpi=500)

tree.plot_tree(clf,

               feature_names = fn, 

               class_names=cn,

               filled = True);

fig.savefig('Titanic_DT.png')
train_mini = train_data.drop('PassengerId', axis=1)

train_mini = train_mini.drop('Age', axis=1)   # hash.remove this if you want to include it

train_mini = train_mini.drop('Fare', axis=1)  # hash.remove this if you want to include it

train_mini = train_mini.drop('SibSp', axis=1) # hash.remove this if you want to include it

train_mini = train_mini.drop('Pclass', axis=1) # hash.remove this if you want to include it



DT_mini_clf = tree.DecisionTreeClassifier(random_state=0)

clf_mini = DT_mini_clf.fit(train_mini, label)



fn=list(train_mini.columns)

cn=["0","1"]

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (2,2), dpi=700)

tree.plot_tree(clf_mini,

               feature_names = fn, 

               class_names=cn,

               filled = True);
 #Decision Tree Score

DT_score = cross_val_score(DT_clf, train_data, label, cv=k_fold, n_jobs=1, scoring='accuracy')

print(DT_score)

round(np.mean(DT_score)*100, 2)
RF_clf = RandomForestClassifier(n_estimators=13)



RF_score = cross_val_score(RF_clf, train_data, label, cv=k_fold, n_jobs=1, scoring='accuracy')

print(RF_score)
 #Random Forest Score

round(np.mean(RF_score)*100, 2)
test.isnull().sum()
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)



test
test.isnull().sum()
test["Fare"] = test.Fare.astype(float)
test.info()
test.drop('Cabin',axis=1,inplace=True) 


test['Fare'] = test['Fare'].fillna(0)



test.isnull().sum()
sex = pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)



test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)



test = pd.concat([test,sex,embark],axis=1)



test
test.info()
xtrain=train_data

ytrain=label

xtest=test
RF=RandomForestClassifier(random_state=1)

PRF=[{'n_estimators':[10,100],'max_depth':[3,6],'criterion':['gini','entropy']}]

GSRF=GridSearchCV(estimator=RF, param_grid=PRF, scoring='accuracy',cv=2)

scores_rf=cross_val_score(GSRF,xtrain,ytrain,scoring='accuracy',cv=5)

np.mean(scores_rf)
model=GSRF.fit(xtrain, ytrain)

pred=model.predict(xtest)




output = pd.DataFrame({'PassengerId': xtest.PassengerId, 'Survived': pred})

output.to_csv('Kamal_submission.csv', index=False)  #change name to own

print("Your submission was successfully saved!")