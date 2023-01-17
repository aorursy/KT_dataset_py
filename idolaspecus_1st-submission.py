import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_raw = pd.read_csv('/kaggle/input/titanic/train.csv')
test_raw = pd.read_csv('/kaggle/input/titanic/test.csv')
alldata = pd.concat([train_raw,test_raw], sort=False, axis=0)
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
alldata['Ticket1'] = (alldata['Ticket'].str[:1]=='1')*1
alldata['Ticket2'] = (alldata['Ticket'].str[:1]=='2')*1
alldata['Ticket3'] = (alldata['Ticket'].str[:1]=='3')*1
alldata['TicketP'] = (alldata['Ticket'].str[:1]=="P")*1
alldata['TicketS'] = (alldata['Ticket'].str[:1]=="S")*1
alldata['TicketC'] = (alldata['Ticket'].str[:1]=="C")*1
alldata['TicketA'] = (alldata['Ticket'].str[:1]=="A")*1

#alldata = alldata.drop(['Ticket','PassengerId'], axis=1)
alldata['Cabin1'] = alldata['Cabin'].str[:1]
alldata['Family']=0
for i in range(0,len(alldata)):
    alldata['Family'].iloc[i] = alldata['Name'].iloc[i][:alldata['Name'].iloc[i].find(',')]
family = pd.DataFrame(alldata['Family'].value_counts())

alldata['NumofFam']=1
for i in range(0,len(alldata)):
    alldata['NumofFam'].iloc[i]=family.loc[alldata['Family'].iloc[i]][0]
alldata['Title']=0
for i in range(0,len(alldata)):
    alldata['Title'].iloc[i] = alldata['Name'].iloc[i][alldata['Name'].iloc[i].find(',')+2:alldata['Name'].iloc[i].find('.')]
(alldata['Embarked']=="Q").sum()
alldata['Survived'].groupby(alldata['Embarked']).mean()
alldata['TitleMr'] = (alldata['Title'].str[:3]=='Mr')*1
alldata['TitleMiss'] = (alldata['Title'].str[:4]=='Miss')*1
alldata['TitleMrs'] = (alldata['Title'].str[:3]=='Mrs')*1
alldata['TitleMaster'] = (alldata['Title'].str[:3]=="Mas")*1
alldata['TitleOther'] = np.abs(alldata['TitleMr']+alldata['TitleMiss']+alldata['TitleMrs']+alldata['TitleMaster']-1)

pd.get_dummies(alldata.drop(['Name','Cabin','Family','Ticket','PassengerId','Title'],axis=1)).corr()['Survived'].sort_values()
alldata2=pd.get_dummies(alldata.drop(['Name','Cabin','Family','Ticket','PassengerId'
                                          ,'Title'],axis=1))
alldata2 = alldata2.fillna(alldata2.median()).drop(['Cabin1_T','Cabin1_G','Cabin1_F'],axis=1)
X = alldata2[:len(train_raw)].drop(['Survived'], axis=1)
Y = alldata2[:len(train_raw)]['Survived']
test = alldata2[len(train_raw):].drop(['Survived'], axis=1)
# grid search
param=[]
# for j in [30,40,50,70,100,200]:
#     for i in [3,5,7,9,11,15,20,None] :
#         rf = RandomForestClassifier(n_estimators=(j*10), max_depth=i, n_jobs=-1)
#         rf.fit(X,Y)
#         print("est=", j*10, "depth=", i, "", rf.score(X,Y))
#         param.append([j,i,rf.score(X,Y)])



from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000,max_depth=None,min_samples_leaf=5,
                            max_features='auto', n_jobs=-1)
rf.fit(X,Y)

from xgboost import XGBClassifier
xgb = XGBClassifier(max_depth=9999, silent=False)
xgb.fit(X,Y)

import lightgbm as lgb
lgb=lgb.LGBMRegressor(n_estimators=110, num_leaves=6)
lgb.fit(X, Y)

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
ridge = RidgeCV(alphas=[8])
ridge.fit(X, Y)

from sklearn.linear_model import Lasso
lasso = LassoCV(alphas=[0.0008], normalize=True)
lasso.fit(X.drop([146], axis=0), Y.drop([146], axis=0))
# for j in [2, 3, 4, 5, 7, 10, 20]:
#     for i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000]:
#         from xgboost import XGBClassifier
#         xgb = XGBClassifier(n_estimators=i, max_depth=j, silent=False)
#         xgb.fit(X,Y)
#         print(i, j, -cross_val_score(xgb, X, Y, n_jobs=-1, cv=5, scoring='neg_mean_squared_error').mean())

xgb = XGBClassifier(n_estimators=400, max_depth=3, silent=False)
xgb.fit(X,Y)

for i in [0.0008]:
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Lasso
    lasso = LassoCV(alphas=[i], max_iter=10000, normalize=True)
    lasso.fit(X.drop([146],axis=0), Y.drop([146],axis=0))
    print(i, -cross_val_score(lasso, X, Y, n_jobs=-1, cv=5,
                              scoring='neg_mean_squared_error').mean())
for i in set(range(0,5))-set([146]):
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
    ridge = RidgeCV(alphas=[0.0008], normalize=True)
    ridge.fit(X, Y)
    MSEs.loc[i]=-cross_val_score(lasso, X.drop([i],axis=0), Y.drop([i],axis=0), 
                                 n_jobs=-1, cv=4, scoring='neg_mean_squared_error').mean()
    if(MSEs.loc[i][0]<0.135):
        print(i, MSEs.loc[i][0])
    if(i%50==0):
        print(i, MSEs.loc[i][0])
lassoX, lassoTrain=lasso.predict(test), lasso.predict(X)
ridgeX, ridgeTrain=ridge.predict(test), ridge.predict(X)
lgbX, lgbTrain=lgb.predict(test), lgb.predict(X)
xgbX, xgbTrain=xgb.predict(test), xgb.predict(X)

XStacking = X
XStacking['lasso']=lassoTrain
XStacking['lgb']=lgbTrain
XStacking['xgb']=xgbTrain

#XStacking['ridge']=ridgeTrain


testStacking = test
testStacking['lasso']=lassoX
testStacking['lgb']=lgbX
testStacking['xgb']=xgbX
#testStacking['ridge']=ridgeX

from sklearn.ensemble import RandomForestClassifier
rf2 = RandomForestClassifier(n_estimators=1000,max_depth=None,min_samples_leaf=5,
                            max_features='auto', n_jobs=-1)
rf2.fit(X,Y)

from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100, n_jobs=-1, max_depth=4)
xgb.fit(X,Y)

-cross_val_score(lgb, X, Y, n_jobs=-1, cv=5, scoring='neg_mean_absolute_error').mean()
from sklearn.model_selection import cross_val_score
for i in [lasso, ridge, xgb, lgb, rf]:
    print(i)
    print(np.sqrt(-cross_val_score(i, X, Y, n_jobs=-1, cv=4, scoring='neg_mean_squared_error' )))
features=pd.DataFrame(np.zeros(len(X.columns)))
features['features']=X.columns
features['importances']=rf2.feature_importances_
features.sort_values(['importances'])
pred=rf2.predict(test)
pred=pred.astype('int')
submission["Survived"] = pred
submission.to_csv("/kaggle/working/submission.csv", index=False)
X['Fare']=np.log(X['Fare']+0.001)
X['Age']=np.log(X['Age']+0.001)

test['Fare']=np.log(test['Fare']+0.001)
test['Age']=np.log(test['Age']+0.001)
X_NN  = pd.concat([X, pd.get_dummies(X[['Sex','Cabin','Embarked','ptitle']])], axis=1)
X_NN = X_NN.drop(['Sex','Cabin','Embarked','ptitle'],axis=1)

test_NN  = pd.concat([test, pd.get_dummies(test[['Sex','Cabin','Embarked','ptitle']])], axis=1)
test_NN = test_NN.drop(['Sex','Cabin','Embarked','ptitle'],axis=1)

test_NN['Cabin_11.0'] = 0
test_NN['Embarked_-1.0'] = 0
print(test_NN.columns)
print(X_NN.columns)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout

model = keras.Sequential([
    layers.Dense(200, activation='relu', input_shape=[len(X_NN.keys())]),
    Dropout(0.0),
    layers.Dense(10, activation='relu'), Dropout(0.1),
    layers.Dense(10, activation='relu'), Dropout(0.1),
    layers.Dense(10, activation='relu'), Dropout(0.1),
    layers.Dense(200, activation='relu'),  Dropout(0.05),
    layers.Dense(200, activation='relu'),  Dropout(0.05), 
    layers.Dense(10, activation='relu'), 
    layers.Dense(5, activation='linear'), 
    layers.Dense(1,activation='sigmoid')])

optimizer = tf.keras.optimizers.Adam(0.005)
model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['binary_accuracy'])

model.fit(X_NN, Y, epochs=20, validation_split = 0, batch_size=100, verbose=0)
loss, val_accuracy = model.evaluate(X_NN, Y, verbose=2)
for i in range(100):
    print(i, (model.predict(X_NN)>(i/100)).sum()-(Y==1).sum())
X_NN.shape, test_NN.shape
pred = (model.predict(test_NN)>0.42)*1

submission["Survived"] = pred
submission.to_csv("/kaggle/working/submission.csv", index=False)