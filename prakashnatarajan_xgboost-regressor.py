# import the libraries

import pandas as pd

import numpy as np

import xgboost as xgb

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
#Load the dataset

df = pd.read_csv(r'../input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv')
#get the shape of the data

df.shape
#get information about the dataframe

df.info()
df['Date'] = pd.to_datetime(df['Date'])
#Visuvalize the relation between oil pirce and Worlt total cases

x=df['Price']

y=df['World_total_cases']

plt.figure(figsize = (20,10));

plt.plot(x,y,'g--')

plt.title('Oil Price  Vs  World_total_cases')

plt.xlabel('Oil Price')

plt.ylabel('Covid_total_cases')
#get the location of the World_total_cases

df.columns.get_loc("World_total_cases")



#selecting the last 5 columns from dataframe

corr_df=df[df.columns[[841,842,843,844,849]]]
# Explore the top 5 rows of the dataset

corr_df.head()
#Finding the coorelation

correlation=corr_df.corr()
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(correlation,vmin=0, vmax=1, annot=True, fmt="g", cmap='coolwarm')
cols = [0,841,842,843,844,849]

data= df[df.columns[cols]]
#Print the top 5 rows

data.head()
data=data.set_index('Date')
#Separate the target variable and rest of the variables using .iloc to subset the data.

X = data.iloc[:,:-1]

y=data.iloc[:,-1]
#convert the dataset into an optimized data structure called Dmatrix that XGBoost supports

data_dmatrix = xgb.DMatrix(data=X,label=y)

#create the train and test set for cross-validation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# instantiate an XGBoost regressor object by calling the XGBRegressor()

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)
#Fit the regressor to the training set and make predictions on the test set

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
#Compute the rmse by invoking the mean_sqaured_error function from sklearn's metrics module.

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse))
#k-fold Cross Validation using XGBoost

params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)


cv_results.head()
print((cv_results["test-rmse-mean"]).tail(1))
#Visualize Feature Importance (features are ordered according to how many times they appear)

import matplotlib.pyplot as plt

xgb.plot_importance(xg_reg)

plt.rcParams['figure.figsize'] = [5, 5]

plt.show()