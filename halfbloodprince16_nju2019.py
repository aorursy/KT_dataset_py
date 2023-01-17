import numpy as np

import pandas as pd 

import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sub = pd.read_csv("../input/samplesubmission (1).csv")
train.head(2)
train.info()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train["penalty"] = le.fit_transform(train["penalty"])
X = train.iloc[:,:-1]

Y= train.iloc[:,[-1]]
#X.head()
from sklearn.feature_selection import f_regression

score = f_regression(X,Y)
score
X.shape[1]
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

import matplotlib.pyplot as plt

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

sfs = SFS(lr, 

          k_features=(1,X.shape[1]), 

          forward=False, # Backward

          floating=False, 

          scoring='neg_mean_squared_error',

          cv=5)

sfs = sfs.fit(X.as_matrix(), Y.as_matrix())
a=sfs.get_metric_dict()

n=[]

o=[]



# Compute the mean of the validation scores

for i in np.arange(1,13):

    n.append(-np.mean(a[i]['cv_scores'])) 

m=np.arange(1,13)



# Plot the Validation scores vs number of features

fig2=plt.plot(m,n)

fig2=plt.title('Mean CV Scores vs No of features')

fig2.figure.savefig('fig2.png', bbox_inches='tight')



print(pd.DataFrame.from_dict(sfs.get_metric_dict(confidence_interval=0.90)).T)



# Get the index of minimum cross validation error

idx = np.argmin(n)

print("No of features=",idx)

#Get the features indices for the best forward fit and convert to list

b=list(a[idx]['feature_idx'])

# Index the column names. 

# Features from backward fit

print("Features selected in bacward fit")

print(X.columns[b])
X_opt = X.iloc[:,[1,4,6,7,8,9]]

from sklearn.model_selection import train_test_split 

Xtrain,Xtest,ytrain,ytest = train_test_split(X_opt,Y,test_size=0.28,random_state=43)
from sklearn.metrics import mean_squared_error as mse
import xgboost as xgb

model = xgb.XGBRegressor(learning_rate =0.017,

                         n_estimators=1000,

                         max_depth=200,

                         min_child_weight=43,

                         gamma=0,

                         subsample=0.8,

                         colsample_bytree=0.8,

                         objective= 'reg:tweedie',

                         nthread=4,

                         scale_pos_weight=1,

                         seed=27

                        )

model.fit(Xtrain,ytrain)

ypred1 = model.predict(Xtest)

print(np.sqrt(mse(ytest,ypred1)))
import lightgbm as lgb

d_train = lgb.Dataset(Xtrain, label=ytrain)

params = {

        'boosting_type': 'gbdt',

        'objective': 'regression',

        'metric': 'rmsle',

        'max_depth': 163, 

        'learning_rate': 0.027,

        'verbose': 0, 

        'min_data': 100,

        'min_depth':143,

        'num_leaves':143

        }

reg = lgb.train(params, d_train, 1000)

ypred2 = reg.predict(Xtest)

print(np.sqrt(mse(ytest,ypred2)))
from sklearn.ensemble import GradientBoostingRegressor 

gb = GradientBoostingRegressor(learning_rate=0.027, 

                              n_estimators=1000, 

                              subsample=1.0, 

                              min_samples_split=27, 

                              min_samples_leaf=63, 

                              min_weight_fraction_leaf=0.0, 

                              max_depth=63)

ypred3 = gb.fit(Xtrain,ytrain).predict(Xtest)

print(np.sqrt(mse(ytest,ypred3)))
from sklearn.ensemble import BaggingRegressor 

br = BaggingRegressor(n_estimators=1000)

ypred4 = br.fit(Xtrain,ytrain).predict(Xtest)

print(np.sqrt(mse(ytest,ypred4)))
from sklearn.ensemble import AdaBoostRegressor

ab = AdaBoostRegressor(n_estimators=5000, learning_rate=0.027)

ypred5 = ab.fit(Xtrain,ytrain).predict(Xtest)

print(np.sqrt(mse(ytest,ypred5)))
stack = pd.DataFrame()

stack["p1"] = model.predict(X_opt)

stack["p2"] = reg.predict(X_opt)

stack["p3"] = gb.predict(X_opt)

stack["p4"] = br.predict(X_opt)

stack["p5"] = ab.predict(X_opt)

stack["time"] = Y
stack.head()
ypred = model.fit(stack.iloc[:,:-1],stack.iloc[:,[-1]])
test["penalty"] = le.fit_transform(test["penalty"])

testX = test.iloc[:,[1,4,6,7,8,9]]
pred = model.fit(X_opt,Y).predict(testX)

stack1 = pd.DataFrame()

stack1["p1"] = model.predict(testX)

stack1["p2"] = reg.predict(testX)

stack1["p3"] = gb.predict(testX)

stack1["p4"] = br.predict(testX)

stack1["p5"] = ab.predict(testX)
pred = model.fit(stack.iloc[:,:-1],stack.iloc[:,[-1]]).predict(stack1)
sub["time"] = pred
sub.to_csv("sub.csv",index=False)