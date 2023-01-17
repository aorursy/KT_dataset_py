import sklearn

import matplotlib.pyplot as plt

%matplotlib inline

import pandas

from sklearn.cross_validation import train_test_split

import numpy

import xgboost as xgb
train=pandas.read_csv("../input/train.csv")

train.head()
del train['id']
training,testing = train_test_split(train,test_size=0.2,random_state=42)

print(training.shape)

print(testing.shape)
Response= training['loss']
print ('Mean of Response Variable'+' '+'is'+' '+ str(numpy.mean(Response)))

print ('Median of Response Variable'+' '+'is'+' '+ str(numpy.median(Response)))

print ('Standard Deviation of Response Variable'+' '+'is'+' '+ str(numpy.std(Response)))
import statsmodels.api as sm

fig=sm.qqplot(Response)
training=training.reset_index(drop=True)

testing = testing.reset_index(drop=True)

training['logloss']=numpy.log(training['loss'])

fig2=sm.qqplot(training['logloss'])
features = training.columns

cat_feature=list(features[0:116])

test=pandas.read_csv("../input/test.csv")

for each in cat_feature:

    training[each]=pandas.factorize(training[each], sort=True)[0]

    testing[each]=pandas.factorize(testing[each],sort=True)[0]

    test[each]=pandas.factorize(test[each],sort=True)[0]
from sklearn.metrics import mean_absolute_error as mae

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor

Predictors= training.ix[:,0:130]

Predictors_test= testing.ix[:,0:130]
Regressors = [LinearRegression(),Lasso(),DecisionTreeRegressor()

              #,RandomForestRegressor(n_estimator=200),

              #GradientBoostingRegressor(learning_rate=0.3,criterion='mae')

             ]

MAE=[]

Model_Name=[]

for reg in Regressors:

    Model=reg.fit(Predictors,training['logloss'])

    Prediction= numpy.exp(Model.predict(Predictors_test))

    eva = mae(testing['loss'],Prediction)

    MAE.append(eva)

    Name=reg.__class__.__name__

    Model_Name.append(Name)

    print('Accuracy of'+ ' '+Name+' '+'is'+' '+str(eva))
MAE.append(1212.42750158)

MAE.append(1183.90923254)

Model_Name.append('RandomForestRegressor')

Model_Name.append('GradientBoostingRegressor')
Index = [1,2,3,4,5]

plt.bar(Index,MAE)

plt.xticks(Index, Model_Name,rotation=45)

plt.ylabel('MAE')

plt.xlabel('Model')

plt.title('MAE of Models')
training_array = numpy.array(Predictors)

testing_array = numpy.array(Predictors_test)
dtrain = xgb.DMatrix(training_array, label=training['logloss'])

dtest = xgb.DMatrix(testing_array)

xgb_params = {

    'seed':0,

    'colsample_bytree': 0.7,

    'subsample': 0.7,

    'learning_rate': 0.075,

    'objective': 'reg:linear',

    'max_depth': 6,

    'min_child_weight': 1,

    'eval_metric': 'mae',

}

xgb_model=xgb.train(xgb_params, dtrain,750,verbose_eval=50)

xgb_pred=numpy.exp(xgb_model.predict(dtest))

print('Accuracy of XGboost model is'+' '+str(mae(testing['loss'],xgb_pred)))
features=Predictors.columns

train['logloss']=numpy.log(train['loss'])

for each in cat_feature:

    train[each]=pandas.factorize(train[each], sort=True)[0]

del test['id']
train_array=numpy.array(train[features])

train_d=xgb.DMatrix(train_array,label=train['logloss'])

test_array=numpy.array(test)

test_d=xgb.DMatrix(test_array)
Final_model=xgb.train(xgb_params, train_d,750,verbose_eval=50)

Prediction_Final=numpy.exp(Final_model.predict(test_d))
submission = pandas.read_csv('../input/sample_submission.csv')

submission.iloc[:, 1] = Prediction_Final

submission.to_csv('sub_xgb.csv', index=None)