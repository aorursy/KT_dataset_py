import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#from sklearn.preprocessing import OneHotEncoder,LabelEncoder, Imputer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectPercentile,SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns',22)
%matplotlib inline


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#### loading the csv's
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sampleSubmission.csv')
#### manipulation of the train data
target = train.iloc[:,-1]
train.drop(['Yield','ID'],axis= 1, inplace=True)
col = train.columns
train[col]= train[col].fillna(train.mode().iloc[0])
#train.isnull().sum()
cols_to_scale = ['AvgAirTemp', 'MinAirTemp','MaxAirTemp', 'AvgTempSkew', 'AvgTempKurt', 'AvgRelHum',
       'AvgRelHumSkew', 'AvgRelHumKurt', 'AvgDewPt', 'AvgDewPtSkew','AvgDewPtKurt', 'AvgPrec', 'AvgPrecSkew', 'AvgPrecKurt', 'AvgWind']
scaler = StandardScaler()
train[cols_to_scale] = scaler.fit_transform(train[cols_to_scale])
train.head()
data = pd.get_dummies(train,drop_first=True)
print("Shape of the data is now: ", data.shape)
X,y  = data, target
xtrain,xtest, ytrain,ytest = train_test_split(X,y,test_size=0.3,shuffle = True,random_state = 20)
model_etr = ExtraTreesRegressor(criterion='mae',n_estimators=50,verbose=1)
model_etr.fit(xtrain,ytrain)

print( "MAE on xtest is : ",mean_absolute_error(ytest, model_etr.predict(xtest)))



#### now enginnering test data...
col = test.columns
test[col]= test[col].fillna(test.mode().iloc[0])
test.drop('ID',inplace=True, axis=1)
testing_data = pd.get_dummies(test,drop_first=True)
testing_data.shape
predicitons = model_etr.predict(testing_data)
submission.Yield = predicitons
submission.to_csv('model_extra_150_85features.csv',index=False)

####################### Neural Network  ############################
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
data.head()
# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(85, input_dim=85, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model
# fix random seed for reproducibility
seed = 20
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X.values, y.values, cv=kfold, n_jobs=1)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# evaluate model with standardized dataset
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold, n_jobs=1)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# define the model
def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(85, input_dim=85, kernel_initializer='normal', activation='relu'))
    model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold, n_jobs=1)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# define wider model
def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return modelb
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs=1)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
