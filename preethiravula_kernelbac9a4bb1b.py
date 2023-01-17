import sklearn

print(sklearn.__version__)
import calendar

import os

from datetime import datetime

from scipy import stats



import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns



from sklearn.preprocessing import OneHotEncoder



from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor





from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_log_error, r2_score



from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import validation_curve

train_bike = pd.read_csv("../input/train.csv")

test_bike = pd.read_csv("../input/test.csv")
train_bike['hour'] = train_bike['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)

train_bike['day'] = train_bike['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').day)

train_bike['weekday'] = train_bike['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())

train_bike['month'] = train_bike['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)

train_bike['year'] = train_bike['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year)

train_bike.columns
cat_columns = ["season","weather","holiday","workingday", "hour", "day", "weekday", "month", "year"]

for features in cat_columns:

    train_bike[features] = train_bike[features].astype("category")

train_bike = pd.get_dummies(train_bike, columns=['season', 'weather', 'holiday', 'workingday'])



train_bike.columns
train_bike.drop(["datetime", "casual", "registered"], 1, inplace=True)

train_bike.shape
#array = train_bike.values

#X_data = array[:,0:20]

#y_data = array[:,21]

y_data = train_bike["count"]

train_bike.drop(["count"], 1, inplace=True)

X_data = train_bike
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, random_state=42, test_size=0.33)

y_train = np.log1p(y_train)

y_val = np.log1p(y_val)

X_train.shape, y_train.shape, X_val.shape, y_val.shape
rf_class = RandomForestRegressor(n_estimators=50,random_state=42)

LR_class = LinearRegression()

KNR_class = KNeighborsRegressor(n_neighbors=2)

GDR_class = GradientBoostingRegressor(random_state=42)

SVR_class = SVR(kernel='linear')

Lasso_class = Lasso(random_state=42, alpha=0.1)

Ridge_class = Ridge(random_state=42)

DTR_class = DecisionTreeRegressor(random_state=42)
kfold = KFold(random_state=42)

print("Random Forests: ")

print(-cross_val_score(rf_class, X_train, y_train, cv=kfold, scoring='neg_mean_squared_log_error'))

rf_class.fit(X_train, y_train)

forecast_value = rf_class.predict(X_val)

print(mean_squared_log_error(y_val, forecast_value)**(1/2))



print("\n\nKNeighborsRegressor:")

print(-cross_val_score(KNR_class, X_train, y_train, cv=kfold, scoring='neg_mean_squared_log_error'))

KNR_class.fit(X_train, y_train)

forecast_value = KNR_class.predict(X_val)

print(mean_squared_log_error(y_val, forecast_value)**(1/2))



print("\n\n:GradientBoostingRegressor")

print(-cross_val_score(GDR_class, X_train, y_train, cv=kfold, scoring='neg_mean_squared_log_error'))

GDR_class.fit(X_train, y_train)

forecast_value = GDR_class.predict(X_val)

print(mean_squared_log_error(y_val, forecast_value)**(1/2))



print("\n\nLinearRegression:")

print(-cross_val_score(LR_class, X_train, y_train, cv=kfold, scoring='neg_mean_squared_log_error'))

LR_class.fit(X_train, y_train)

forecast_value = LR_class.predict(X_val)

print(mean_squared_log_error(y_val, forecast_value)**(1/2))



print("\n\nLasso:")

print(-cross_val_score(Lasso_class, X_train, y_train, cv=kfold, scoring='neg_mean_squared_log_error'))

Lasso_class.fit(X_train, y_train)

forecast_value = Lasso_class.predict(X_val)

print(mean_squared_log_error(y_val, forecast_value)**(1/2))



print("\n\nRidge:")

print(-cross_val_score(Ridge_class, X_train, y_train, cv=kfold, scoring='neg_mean_squared_log_error'))

Ridge_class.fit(X_train, y_train)

forecast_value = Ridge_class.predict(X_val)

print(mean_squared_log_error(y_val, forecast_value)**(1/2))



print("\n\nDecisionTreeRegressor:")

print(-cross_val_score(DTR_class, X_train, y_train, cv=kfold, scoring='neg_mean_squared_log_error'))

DTR_class.fit(X_train, y_train)

forecast_value = DTR_class.predict(X_val)

print(mean_squared_log_error(y_val, forecast_value)**(1/2))





from sklearn.datasets import load_digits

from sklearn.model_selection import learning_curve

cv = ShuffleSplit(n_splits=100, test_size=0.33, random_state=0)
def Learning_curve_model(X, Y, model, cv, train_sizes):



    plt.figure()

    plt.title("Learning curve")

    plt.xlabel("Training examples")

    plt.ylabel("Score")





    train_sizes, train_scores, test_scores = learning_curve(model, X, Y, cv=cv, n_jobs=4, train_sizes=train_sizes)



    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std  = np.std(train_scores, axis=1)

    test_scores_mean  = np.mean(test_scores, axis=1)

    test_scores_std   = np.std(test_scores, axis=1)

    plt.grid()

    

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

                     

    plt.legend(loc="best")

    return plt
#train_size=np.linspace(.1, 1.0, 15)

# uncomment the below for plotting learning curve

#Learning_curve_model(X_data, y_data, RandomForestRegressor(n_estimators=50, random_state=42), cv, train_size)

#plt.savefig('Learning_curveV2.png')
rf = RandomForestRegressor(n_estimators=50, random_state=42)

y_data = np.log1p(y_data)

rf.fit(X_data,y_data)
test_bike['hour'] = test_bike['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)

test_bike['day'] = test_bike['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').day)

test_bike['weekday'] = test_bike['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())

test_bike['month'] = test_bike['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)

test_bike['year'] = test_bike['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year)

for features in cat_columns:

    test_bike[features] = test_bike[features].astype("category")

test_bike = pd.get_dummies(test_bike, columns=['season', 'weather', 'holiday', 'workingday'])

test_bike.drop(["datetime"], 1, inplace=True)

test_bike.shape
preds = rf.predict(test_bike)
submission = pd.read_csv('../input/sampleSubmission.csv')

submission.head()
submission["count"] = np.expm1(preds)

submission.head()
#submission.to_csv("Final_output.csv", index=False)