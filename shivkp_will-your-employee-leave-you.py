import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
def num_val(data):
    for column in data.columns:
        if data[column].dtype == type(object):
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
    return data
tr_data = pd.read_csv("../input/hacker-earth-will-your-employees-leave-you/Train.csv")
tr_data
#train_data = tr_data.drop(['VAR6','VAR5','VAR1','VAR2','Work_Life_balance','Post_Level','Unit','Hometown','Education_Level','Age','Employee_ID','Relationship_Status','Compensation_and_Benefits','Gender'], axis=1)
train_data = tr_data.drop(['Employee_ID','Relationship_Status','Gender','Age'], axis=1)

train_data
from scipy.stats import norm 
corr_m = train_data.corr()

f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corr_m, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
print(corr_m['Attrition_rate'])
train_data = num_val(train_data)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imput = imputer.fit(train_data)
train_data = imput.transform(train_data)
train_data
X = train_data[:,:-1]
Y = train_data[:,-1].reshape(-1,1)
X,Y
print(X.shape,Y.shape)
test = pd.read_csv("../input/hacker-earth-will-your-employees-leave-you/Test.csv")
test
#X_test = test.drop(['VAR6','VAR5','Work_Life_balance','Post_Level','Unit','Hometown','Education_Level','Age','Employee_ID','Relationship_Status','Gender','Compensation_and_Benefits','VAR1','VAR2'],axis=1)
X_test = test.drop(['Employee_ID','Gender','Relationship_Status','Age'],axis=1)

X_test
X_test = num_val(X_test)

imput = imputer.fit(X_test)
X_test = imput.transform(X_test)
X_test.shape
from sklearn.preprocessing import StandardScaler, PolynomialFeatures,MinMaxScaler

stdx = MinMaxScaler()
stdx.fit(X)
X = stdx.transform(X)

stdy = StandardScaler()
stdy.fit(Y)
Y = stdy.transform(Y)
X, Y
X_test = stdx.transform(X_test)
X_test
import keras
import keras.backend as kb
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from keras.layers import Dropout
model = Sequential()
model.add(Dense(32, input_dim=19, kernel_initializer="uniform", activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 32, kernel_initializer="uniform", activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 1, kernel_initializer="uniform"))
model.summary()

model.compile(loss='mse', optimizer = 'adam', metrics=['mse','mae'])
epochs_hist = model.fit(X, Y, epochs=20, batch_size=20, validation_split=0.99)
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])
y_ker = model.predict(X_test)
y_ker = stdy.inverse_transform(y_ker)
y_ker
import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',colsample_bytree = 1, learning_rate = 0.0335, max_depth = 5, alpha = 1, n_estimators = 100)
#xg_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.35, gamma=0,subsample=0.75, colsample_bytree=0.4, max_depth=7)
xg_reg.fit(X,Y)
from xgboost import plot_importance
import matplotlib.pyplot as plt

# plot feature importance
plot_importance(xg_reg)
plt.show()
#xg_reg.feature_importances_
y_xg = xg_reg.predict(X_test)
y_xg = stdy.inverse_transform(y_xg)
y_xg
import statsmodels.api as sm
sm_model = sm.GLSAR(Y,X, rho=18).fit()
sm_model.params

y_sm = sm_model.predict(X_test)
y_sm = stdy.inverse_transform(y_sm)
y_sm
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=20)
rf.fit(X,Y.ravel())

y_rf = rf.predict(X_test)
y_rf = stdy.inverse_transform(y_rf)
y_rf
from sklearn.linear_model import RidgeCV
Rid = RidgeCV(alphas=np.arange(0.1,100,1), fit_intercept=True)
Rid.fit(X,Y)

y_rid = Rid.predict(X_test)
#y_rid = stdy.inverse_transform(y_rid)
y_rid = stdy.inverse_transform(y_rid)
y_rid
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import LinearRegression
#compute_score=True
lr = BayesianRidge(compute_score=True)
lr.fit(X,Y.ravel())

y_lr = lr.predict(X_test)
y_lr = stdy.inverse_transform(y_lr)
y_lr
from sklearn.svm import SVR
regressor = SVR(kernel='rbf', gamma='scale', tol=0.0000000000001, epsilon=0.2)
regressor.fit(X, Y.ravel())

y_rg = regressor.predict(X_test)
y_rg = stdy.inverse_transform(y_rg)
y_rg
data = pd.DataFrame()
data['Employee_ID'] = test['Employee_ID']
data['Attrition_rate'] = y_lr

data = data.to_csv('Emp_submission.csv',mode='w',index=False)
