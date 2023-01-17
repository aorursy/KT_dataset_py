import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



pd.set_option('display.max_columns', 100)
#getting data

test = pd.read_csv('../input/eval-lab-1-f464-v2/test.csv')

df = pd.read_csv('../input/eval-lab-1-f464-v2/train.csv')

df.fillna(value=df.median(),inplace=True)

test.fillna(value=test.median(),inplace=True)
#Constructing 4 sets of data, (x,y)-> (x_test,~); (x_train,y_train);(x_cv,y_cv)

encoded = pd.get_dummies(df['type'])

x = df.copy()

y = df["rating"].copy()

x=x.drop(['rating','type'],axis=1);



from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()#MinMaxScaler()

X_scaled = scaler.fit_transform(x)

x = np.concatenate([X_scaled,encoded.values],axis=1)



from sklearn.model_selection import train_test_split

x_train,x_cv,y_train,y_cv = train_test_split(x,y,test_size=0.33)#random_state=42)

np.delete(x_train,0,axis=1);

np.delete(x_cv,0,axis=1);

np.delete(x,0,axis=1);

[x_train.shape, y_train.shape, x.shape, y.shape, x_cv.shape, y_cv.shape]
encodedt = pd.get_dummies(test['type'])

x_test = test.copy()

x_test=x_test.drop(['type'],axis=1);

from sklearn.preprocessing import MinMaxScaler



#scaler = StandardScaler()#MinMaxScaler()#StandardScaler()#

X_scaledt = scaler.fit_transform(x_test)

x_test = np.concatenate([X_scaledt,encodedt.values],axis=1)

np.delete(x_test,0,axis=1);

x_test.shape
y_train.value_counts()
# ncrese n_estimators #number of trees

 

#  reduce max_features to 5-7

#  max_depth : start with 5- 10 and increase

#  min_samples_leaf : >1

from sklearn.ensemble import RandomForestClassifier

from sklearn import ensemble

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

# regressor = MLPRegressor(hidden_layer_sizes=(128,128), activation='logistic', solver='adam',

#                                     alpha=0.0001, batch_size='auto', learning_rate='constant', 

#                                     learning_rate_init=0.00001, power_t=0.5, max_iter=10000, shuffle=True,

#                                     random_state=None, tol=0.0001, verbose=True, warm_start=False, 

#                                     momentum=0.9, nesterovs_momentum=True, early_stopping=True, 

#                                     validation_fraction=0.2, n_iter_no_change=50)

# regressor = SVR(kernel='rbf', C=8, gamma='auto', degree=3, epsilon=.1,

#                coef0=1)

# regressor = RandomForestClassifier(max_depth=10, max_features=10,max_leaf_nodes=100,

#                                   min_impurity_decrease=0.0,

#                                   min_impurity_split=None, min_samples_leaf=2, 

#                                    n_estimators=1000, n_jobs=-1,

#                                   oob_score=False, random_state=None, verbose=0,warm_start=False)

regressor = ensemble.ExtraTreesRegressor(max_depth=10, max_features=10,max_leaf_nodes=100,

                                  min_impurity_decrease=0.0,

                                  min_impurity_split=None, min_samples_leaf=2, 

                                   n_estimators=1000, n_jobs=-1,

                                  oob_score=False, random_state=None, verbose=0,warm_start=False,)

# regressor = ensemble.GradientBoostingRegressor(n_estimators= 500, max_depth= 4, min_samples_split= 2,learning_rate= 0.05, loss='ls')

regressor.fit(x_train,y_train)
from sklearn.ensemble import RandomForestClassifier

from sklearn import ensemble



# regressor = ensemble.GradientBoostingClassifier(min_samples_leaf = 0.1, subsample = 0.5, learning_rate = 0.05, max_depth = 2)

# parameters = {

#     "loss":["deviance"],

#     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],

#     "min_samples_split": np.linspace(0.1, 0.5, 12),

#     "min_samples_leaf": np.linspace(0.1, 0.5, 12),

#     "max_depth":[3,5,8],

#     "max_features":["log2","sqrt"],

#     "criterion": ["friedman_mse",  "mae"],

#     "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],

#     "n_estimators":[10]

#     }



# regressor = GridSearchCV(clf, parameters, cv=10, n_jobs=-1)

# regressor.fit(x_train,y_train)
#Calculating y_pred from x_cv

y_pred_cv=regressor.predict(x_cv)

y_pred_train=regressor.predict(x_train)
#rounding y_cv to int between 0 - 6

y_pred_cv=np.rint(y_pred_cv);

y_pred_cv.astype(np.uint64)

y_pred_cv= [0 if x <= 0 else 6 if x > 6 else x for x in y_pred_cv]



#rounding y_train to int between 0 - 6

y_pred_train=np.rint(y_pred_train);

y_pred_train.astype(np.uint64)

y_pred_train= [0 if x <= 0 else 6 if x > 6 else x for x in y_pred_train]
#cv rmse

from sklearn.metrics import mean_absolute_error

rmse_cv = mean_absolute_error(y_pred_cv,y_cv)

#Train rmse

rmse_train = mean_absolute_error(y_pred_train,y_train)

[rmse_train,rmse_cv]
plt.scatter(y_pred_cv,y_cv,alpha = 0.1)
y_pred1=y_pred1+y_pred_cv



plt.scatter(y_pred1,y_cv)
mean_absolute_error(y_pred1,y_cv)
#Training on entire dataset x,y

regressor.fit(x,y)
#Train rmse for entire data

rmse = mean_absolute_error(regressor.predict(x),y)
#Calculating y_test from x_test

y_test=regressor.predict(x_test)
#rounding y to int between 0 - 6

y_test=np.rint(y_test);

y_test.astype(np.uint64)

y_test= [0 if x <= 0 else 6 if x > 6 else x for x in y_test]


rmse
rmse_cv
rmse_train
plt.scatter(y_pred_cv,y_cv,alpha = 0.1)
y_cv.value_counts()
y_test
#storing y_test in reuired format

ID = test['id']

#y_test= y_test.reshape(len(y_test),1)

ans = pd.concat([ID,pd.DataFrame(y_test)],axis=1)
#check the things

ans.dtypes
ans=ans.astype('int32')

ans[0].value_counts()
#store in csv

ans.to_csv("submitrs.csv",index=None,header=["id","rating"])
rmse_cv
rmse_train
rmse
#Gradient Boosting

from sklearn import ensemble

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)



clf.fit(x_train, y_train)