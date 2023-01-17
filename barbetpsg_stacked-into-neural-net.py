# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.model_selection import KFold



from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop

from keras.callbacks import ReduceLROnPlateau



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



pid = test['PassengerId']



train.head()
# test on only Pclass and gender

from sklearn.model_selection import cross_val_score





train_ex = train.copy()

train_ex['is_male'] = train_ex['Sex'].apply(lambda x: 1 if x == 'male' else 0)



X_train = train_ex[['Pclass','is_male']].values

y_train = train_ex['Survived'].values.ravel()



clf = LogisticRegression(C=1, penalty='l1')



print(cross_val_score(clf, X_train, y_train, cv=5))
train.isnull().sum(axis=0)
##Let's do some feature engineering



full_data = [train, test]



# filling age nulls

for ds in full_data:

    age_avg = ds['Age'].mean()

    age_std = ds['Age'].std()

    null_count = ds['Age'].isnull().sum()

    rand_age_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=null_count)

    ds['Age'][np.isnan(ds['Age'])] = rand_age_list

    ds['Age'] = ds['Age'].astype(int)

    

train['CatAge'] = pd.qcut(train['Age'],5)



# making categorical fare field    

for ds in full_data:

    ds['Fare'] = ds['Fare'].fillna(train['Fare'].median())



train['CatFare'] = pd.qcut(train['Fare'], 5)

    

# making family size and isalone bool features

for ds in full_data:

    ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1



for ds in full_data:

    ds['IsAlone'] = 0

    ds.loc[ds['FamilySize']==1,'IsAlone'] = 1
print(train[['CatFare','Survived']].groupby('CatFare', as_index=False).mean())
print(train[['CatAge','Survived']].groupby('CatAge', as_index=False).mean())
# lets map values to appropriate variables



for ds in full_data:

    ds['Sex'] = ds['Sex'].map({'male': 0, 'female': 1}).astype(int)

    

    ds.loc[ds['Age'] <= 19, 'Age'] = 0

    ds.loc[(ds['Age'] > 19) & (ds['Age'] <= 25), 'Age'] = 1

    ds.loc[(ds['Age'] > 25) & (ds['Age'] <= 31), 'Age'] = 2

    ds.loc[(ds['Age'] > 31) & (ds['Age'] <= 40), 'Age'] = 3

    ds.loc[ds['Age'] > 40, 'Age'] = 4

    ds['Age'] = ds['Age'].astype(int)

    

    ds.loc[ds['Fare'] <= 7.854, 'Fare'] = 0

    ds.loc[(ds['Fare'] <= 10.5) & (ds['Fare'] > 7.854), 'Fare'] = 1

    ds.loc[(ds['Fare'] <= 21.679) & (ds['Fare'] > 10.5), 'Fare'] = 2

    ds.loc[(ds['Fare'] <= 39.688) & (ds['Fare'] > 21.679), 'Fare'] = 3

    ds.loc[ds['Fare'] > 39.688, 'Fare'] = 4

    ds['Fare'] = ds['Fare'].astype(int)
train.drop(columns=['CatAge', 'CatFare', 'Name', 'PassengerId', 'Cabin', 'Ticket', 'Parch', 'SibSp'], inplace=True)

test.drop(columns=['Name', 'PassengerId', 'Cabin', 'Ticket', 'Parch', 'SibSp'], inplace=True)
train.head()
test.head()
for ds in full_data:

    ds['Embarked'].fillna('S', inplace=True)

    ds['Embarked'] = ds['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
X_train = train.drop(columns=['Survived']).values

y_train = train['Survived'].ravel()

X_test = test.values
kf = KFold(n_splits=5, random_state=69)

kf.get_n_splits()
# create function that can get out of fold predictions



def get_oof(clf, X_train, y_train, X_test, kf):

    train_oof = np.zeros((X_train.shape[0],))

    test_oof = np.zeros((X_test.shape[0],))

    test_oof_kf = np.zeros((kf.get_n_splits(), X_test.shape[0]))

    

    for i, (train_idx, test_idx) in enumerate(kf.split(X_train)):

        X_tr = X_train[train_idx]

        y_tr = y_train[train_idx]

        X_te = X_train[test_idx]

        

        clf.fit(X_tr,y_tr)

        

        train_oof[test_idx] = clf.predict(X_te)

        test_oof_kf[i,:] = clf.predict(X_test)

        

    test_oof[:] = test_oof_kf.mean(axis=0)

    

    return train_oof.reshape(-1,1), test_oof.reshape(-1,1)
colormap = plt.cm.RdBu

plt.figure(figsize=(8,8))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=False)
# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}



# Logistic Regression Parameters

logreg_params = {

    'solver': 'saga',

    'l1_ratio': 0.75,

    'penalty': 'elasticnet',

    'C': 0.05

}



# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }





svc = SVC(**svc_params)

logreg = LogisticRegression(**logreg_params)

rf = RandomForestClassifier(**rf_params)

adb = AdaBoostClassifier(**ada_params)

gdb = GradientBoostingClassifier(**gb_params)
svc_train_oof, svc_test_oof = get_oof(svc, X_train, y_train, X_test, kf)

logreg_train_oof, logreg_test_oof = get_oof(logreg, X_train, y_train, X_test, kf)

rf_train_oof, rf_test_oof = get_oof(rf, X_train, y_train, X_test, kf)

adb_train_oof, adb_test_oof = get_oof(adb, X_train, y_train, X_test, kf)

gdb_train_oof, gdb_test_oof = get_oof(gdb, X_train, y_train, X_test, kf)
first_level_train = pd.DataFrame({'LinearSVC': svc_train_oof.ravel(),

                                  'LogisticRegression': logreg_train_oof.ravel(),

                                  'RandomForest': rf_train_oof.ravel(),

                                  'AdaBoost': adb_train_oof.ravel(),

                                  'GradientBoost': gdb_train_oof.ravel()

                                 })



first_level_train.head()
colormap = plt.cm.RdBu

plt.figure(figsize=(8,8))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(first_level_train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=False)
X_train_f = np.concatenate((svc_train_oof, logreg_train_oof, rf_train_oof, adb_train_oof, gdb_train_oof), axis=1)

X_test_f = np.concatenate((svc_test_oof, logreg_test_oof, rf_test_oof, adb_test_oof, gdb_test_oof), axis=1)
final = Sequential()



final.add(Dense(50, activation='relu', input_shape=(5,)))

final.add(Dropout(0.2))



final.add(Dense(50, activation='relu'))

final.add(Dropout(0.2))



final.add(Dense(1, activation='sigmoid'))



opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

lrreduce = ReduceLROnPlateau(monitor='acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)



final.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])



final.fit(X_train_f, y_train, validation_split=0.1, verbose=1, epochs=100, batch_size=50, callbacks=[lrreduce])
import xgboost as xgb



gbm = xgb.XGBClassifier(

    #learning_rate = 0.02,

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(X_train_f, y_train)

out = final.predict(X_test_f)



print(out.shape)
StackingSubmission = pd.DataFrame({ 'PassengerId': pid,

                            'Survived': out.ravel()})

StackingSubmission['Survived'] = StackingSubmission['Survived'].apply(lambda x: 1 if x>0.50 else 0)

StackingSubmission.to_csv("StackingSubmission.csv", index=False)
StackingSubmission.head()