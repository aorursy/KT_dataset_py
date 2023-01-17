import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import StratifiedKFold as SKF

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score as AUC

from sklearn.model_selection import cross_val_score





train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
#adding a column to identify whether a row comes from train or not

test['is_train'] = 0

train['is_train'] = 1 
train['Fare'].fillna(train['Fare'].mean(), inplace=True)

test['Fare'].fillna(train['Fare'].mean(), inplace=True)



train['Age'].fillna(train['Age'].mean(), inplace = True)

test['Age'].fillna(train['Age'].mean(), inplace = True)



train['Sex'].replace(['male','female'],[0,1],inplace=True)

test['Sex'].replace(['male','female'],[0,1],inplace=True)
train['origin'] = 0

test['origin'] = 1

train = train.drop(['Survived','PassengerId','Cabin', 'Ticket','Name', 'Embarked'],axis=1) #droping target variable

test = test.drop(['PassengerId','Cabin', 'Ticket','Name', 'Embarked'],axis=1)



combi = train.append(test)

y = combi['origin']

combi.drop('origin',axis=1,inplace=True)
m = RandomForestClassifier(n_jobs=-1, max_depth=5, min_samples_leaf = 5)
drop_list = []

for i in combi.columns:

    score = cross_val_score(m,pd.DataFrame(combi[i]),y,cv=2,scoring='roc_auc')

    if (np.mean(score) > 0.8):

        drop_list.append(i)

        print(i,np.mean(score))
#combining test and train data

df_combine = pd.concat([train, test], axis=0, ignore_index=True)

#dropping ‘target’ column as it is not present in the test

df_combine = df_combine.drop(['origin'], axis =1)

y = df_combine['is_train'].values #labels

x = df_combine.drop('is_train', axis=1).values #covariates or our independent variables
# m = RandomForestClassifier(n_jobs=-1, max_depth=5, min_samples_leaf = 5)

shape = y.shape[0],9

predictions = np.zeros(shape) #creating an empty prediction array
skf = SKF(n_splits=9, shuffle=True, random_state=100)

for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):

    X_train, X_test = x[train_idx], x[test_idx]

    y_train, y_test = y[train_idx], y[test_idx]



    m.fit(X_train, y_train)

    probs = m.predict_proba(X_test)[:, 1] #calculating the probability

    predictions[test_idx,fold] = probs
pred_mean = predictions.mean(axis=1)
print('ROC-AUC for train and test distributions:', AUC(y, pred_mean))