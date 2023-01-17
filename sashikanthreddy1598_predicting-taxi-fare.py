# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing libraries:

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_palette("GnBu_d")

sns.set_style('whitegrid')
#Read and Import data:

data = pd.read_csv('../input/train.csv')
#Head 0f Data

data.head()
#Dealing with null values:

data.dropna(subset = ['pickup_longitude', 'pickup_latitude', 'dropoff_latitude', 'dropoff_longitude'], inplace= True)

data.fillna(value = 0, inplace= True)

data.isnull().sum()
#Descriptive Stats:

data.describe()
data = data[data['dropoff_latitude']!=0]

data = data[data['dropoff_longitude']!=0]

                                                #Removing zeros from latitudes and longitudes

data = data[data['pickup_latitude']!=0]

data = data[data['pickup_longitude']!=0]
#Calculating the distance from latitudes and longitudes

def haversine_distance(lat1, long1, lat2, long2):

    dat = [data]

    for i in dat:

        R = 6371  #radius of earth in kilometers

        x1 = np.radians(i[lat1])

        x2 = np.radians(i[lat2])

        delta_x1 = np.radians(i[lat2]-i[lat1])

        delta_lambda = np.radians(i[long2]-i[long1])

 #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)

        a = np.sin(delta_x1 / 2.0) ** 2 + np.cos(x1) * np.cos(x2) * np.sin(delta_lambda / 2.0) ** 2

 #c = 2 * atan2( √a, √(1−a) )

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

 #d = R*c

        d = (R * c) #in kilometers

        i['Distance'] = d

haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')

#Checking the head again:

data.head()
plt.scatter(x=data['Distance'],y=data['fare_amount'], alpha= .2)

plt.xlabel("Trip Distance")

plt.ylabel("Fare Amount")

plt.title("Trip Distance vs Fare Amount")
sns.kdeplot(data['Distance'].values).set_title("Distribution of Trip Distance")
data[data['Distance']>100]
#Removing outliers: (probably wrong data point....)

data = data[data['Distance']<100]

data = data[data['mta_tax'] > 0]
sns.kdeplot(np.log(data['Distance'].values)).set_title("Distribution of Trip Distance (log scale)")
plt.scatter(x=data['Distance'],y=data['fare_amount'], alpha= .2)

plt.xlabel("Trip Distance")

plt.ylabel("Fare Amount")

plt.title("Trip Distance vs Fare Amount")
#Converting the pickup and dropoff time to datetime object:

data['pickup_datetime']=pd.to_datetime(data['pickup_datetime'])



data['dropoff_datetime']=pd.to_datetime(data['dropoff_datetime'])



#Calculating Trip Duration:

data['duration'] = data['dropoff_datetime'] - data['pickup_datetime']



#Getting the seconds from datetime object

data['duration'] = data['duration'].dt.total_seconds()

#Coverting duration into minutes:

data['duration'] = data['duration'] / 60

data.head()

data[data['duration']>200]
data = data[data['duration']<200]
#Adding back the target column:

#data['fare_amount'] = temp

data.head()
plt.figure(figsize=(8,5))

sns.kdeplot(np.log(data['fare_amount'].values)).set_title("Distribution of fare amount (log scale)")
sns.kdeplot(np.log(data['Distance'].values)).set_title("Distribution of Trip Distance (log scale)")
plt.scatter(x=data['Distance'],y=data['fare_amount'], alpha= .2)

plt.xlabel("Trip Distance")

plt.ylabel("Fare Amount")

plt.title("Trip Distance vs Fare Amount")
#print("Avg trip distance (in miles) when there are zero passengers",np.mean(train.loc[train['passenger_count']==0,'Distance'].values))
#Dropping unnecessary columns:

data.drop(['TID','new_user', 'store_and_fwd_flag', 'payment_type', 'mta_tax'],axis = 1, inplace=True)
data.head()
#Dropping location and time columns:

data.drop(['dropoff_longitude', 'dropoff_latitude','pickup_longitude','pickup_latitude','pickup_datetime',

       'dropoff_datetime'], axis = 1 , inplace= True)
#Heatmap of correlation matrix:

plt.figure(figsize= (12,7))

sns.heatmap(data.corr(), annot= True, cmap = 'Reds')
g = trips_year_fareamount=data.groupby(['passenger_count'])['fare_amount'].mean().reset_index().rename(columns={'fare_amount':'avg_fare_amount'})
sns.barplot(x = 'passenger_count', y = 'avg_fare_amount',data=g).set_title("Avg Fare Amount for passenger count")
#Creating dummies for categorical columns:

cat_cols = ['vendor_id']

data[cat_cols] = data[cat_cols].apply(lambda x: x.astype('category'))

data = pd.get_dummies(data, columns = cat_cols, drop_first= True)
#Writing the cleaned data to a csv file



data.to_csv('train_cleaned', index = False)
from sklearn.model_selection import train_test_split

y = data['fare_amount']

X = data.drop(['fare_amount'],axis =1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state =1) 
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Linear Regression Model:

lm = LinearRegression()

#Fitting the Model on train:

lm.fit(X_train, y_train)

#Predicting the model on train:

trainpred = lm.predict(X_train)

#Predicting the model on validation:

testpred = lm.predict(X_test)

print('---------MODEL-METRICS-----------')

print('MAE-Train:\t',mean_absolute_error(y_train, trainpred)** .5)

print('MAE-Test:\t',mean_absolute_error(y_test, testpred)** .5)

print('RMSE-Train:\t',mean_squared_error(y_train, trainpred)** .5)

print('RMSE-Test:\t',mean_squared_error(y_test, testpred)** .5)

print('R2-Train:\t',r2_score(y_train, trainpred))

print('R2-Test:\t',r2_score(y_test, testpred))
plt.scatter(y_test,testpred, alpha = .2)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
sns.distplot(np.log(y_test - testpred),bins = 20).set_title("Distribution of Residuals (log scale)")
from sklearn.linear_model import RidgeCV



#Ridge Regression

Ridge_cv=RidgeCV(alphas=[0.1,1,10,100],cv=10)

Ridge_cv.fit(X_train,y_train)



trainpred_rcv=Ridge_cv.predict(X_train)

testpred_rcv=Ridge_cv.predict(X_test)



print('---------MODEL-METRICS-----------')

print('RMSE-Train:\t',mean_squared_error(y_train, trainpred_rcv)** .5)

print('RMSE-Validation:\t',mean_squared_error(y_test, testpred_rcv)** .5)

print('R2-Train:\t',r2_score(y_train, trainpred_rcv))

print('R2-Validation:\t',r2_score(y_test, testpred_rcv))
from sklearn.linear_model import LassoCV



#Lasso Regression:

Lasso_cv=LassoCV(cv=10)

Lasso_cv.fit(X_train,y_train)



#Predicting the model:

trainpred_lcv=Lasso_cv.predict(X_train)

testpred_lcv=Lasso_cv.predict(X_test)



print('---------MODEL-METRICS-----------')

print('RMSE-Train:\t',mean_squared_error(y_train, trainpred_lcv)** .5)

print('RMSE-Test:\t',mean_squared_error(y_test, testpred_lcv)** .5)

print('R2-Train:\t',r2_score(y_train, trainpred_lcv))

print('R2-Test:\t',r2_score(y_test, testpred_lcv))
from sklearn.linear_model import ElasticNetCV





Elas_cv=ElasticNetCV(l1_ratio=[0.01,0.03,0.1,0.8,0.9],random_state=1,cv=10)

Elas_cv.fit(X_train,y_train)



trainpred_ecv=Elas_cv.predict(X_train)



testpred_ecv=Elas_cv.predict(X_test)



print('---------MODEL-METRICS-----------')

print('RMSE-Train:\t',mean_squared_error(y_train, trainpred_ecv)** .5)

print('RMSE-Validation:\t',mean_squared_error(y_test, testpred_ecv)** .5)

print('R2-Train:\t',r2_score(y_train, trainpred_ecv))

print('R2-Validation:\t',r2_score(y_test, testpred_ecv))
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



#Decision Tree Regression:

dtr=DecisionTreeRegressor()

params = {"min_samples_leaf": [2,5,7,10,12,15],"max_depth": [3,5,7,9,11,13,15]}

#params = {"n_neighbors": [1],"metric": ["euclidean", "cityblock"]}



grid = GridSearchCV(dtr,param_grid=params,scoring="neg_mean_squared_error",cv=10)

grid.fit(X_train,y_train)



grid.best_estimator_



#Decision Tree Regression- Metrics:

trainpred_dtr=grid.predict(X_train)

testpred_dtr=grid.predict(X_test)

print('---------MODEL-METRICS-----------')

print('RMSE-Train:\t',mean_squared_error(y_train, trainpred_dtr)** .5)

print('RMSE-Validation:\t',mean_squared_error(y_test, testpred_dtr)** .5)

print('R2-Train:\t',r2_score(y_train, trainpred_dtr))

print('R2-Validation:\t',r2_score(y_test, testpred_dtr))
#Learning Curves:

from sklearn.model_selection import learning_curve







train_sizes, train_scores, validation_scores = learning_curve(

estimator = Lasso_cv,

X = X,

y = y, train_sizes = np.array([.1, .2, 0.30, .4, 0.50, 0.60, .7, .8, .9]), cv = 10,

scoring = 'neg_mean_squared_error')



print('Training scores:\n\n', train_scores)

print('\n', '-' * 70) # separator to make the output easy to read

print('\nValidation scores:\n\n', validation_scores)
train_scores_mean = -train_scores.mean(axis = 1)

validation_scores_mean = -validation_scores.mean(axis = 1)

print('Mean training scores\n\n', pd.Series(train_scores_mean))

print('\n', '-' * 20) # separator

print('\nMean validation scores\n\n',pd.Series(validation_scores_mean))
import matplotlib.pyplot as plt



plt.style.use('seaborn')

plt.plot(train_sizes, train_scores_mean, label = 'Training error')

plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('MSE', fontsize = 14)

plt.xlabel('Training set size', fontsize = 14)

plt.title('Learning curves for a linear regression model', fontsize = 18, y = 1.03)

plt.legend()

plt.ylim(2,10)
#MEtrics Dataframe

data=[['Linear Regression','Train',r2_score(y_train, trainpred),mean_squared_error(y_train, trainpred)**.5],['Linear Regression','Test',r2_score(y_train, trainpred),(mean_squared_error(y_test, testpred))** .5]]

eval_metrics_df=pd.DataFrame(data,columns=['Model','Train_Test','R-Squared','RMSE'])

eval_metrics_df=eval_metrics_df.append(pd.DataFrame([['Elastinet Regression','Train',r2_score(y_train, trainpred_ecv),(mean_squared_error(y_train, trainpred_ecv))** .5],['Elastinet Regression','Test',r2_score(y_train, trainpred_ecv),(mean_squared_error(y_test, testpred_ecv))** .5]],columns=['Model','Train_Test','R-Squared','RMSE']))

eval_metrics_df=eval_metrics_df.append(pd.DataFrame([['Ridge Regression','Train',r2_score(y_train, trainpred_rcv),(mean_squared_error(y_train, trainpred_rcv))** .5],['Ridge Regression','Test',r2_score(y_train, trainpred_rcv),(mean_squared_error(y_test, testpred_rcv))** .5]],columns=['Model','Train_Test','R-Squared','RMSE']))

eval_metrics_df=eval_metrics_df.append(pd.DataFrame([['Lasso Regression','Train',r2_score(y_train, trainpred_lcv),(mean_squared_error(y_train, trainpred_lcv))** .5],['Lasso Regression','Test',r2_score(y_train, trainpred_lcv),(mean_squared_error(y_test, testpred_lcv))** .5]],columns=['Model','Train_Test','R-Squared','RMSE']))

eval_metrics_df=eval_metrics_df.append(pd.DataFrame([['DT Regression','Train',r2_score(y_train, trainpred_dtr),(mean_squared_error(y_train, trainpred_dtr))** .5],['DT Regression','Test',r2_score(y_train, trainpred_dtr),(mean_squared_error(y_test, testpred_dtr))** .5]],columns=['Model','Train_Test','R-Squared','RMSE']))
#Model RMSE comparison plot:

plt.figure(figsize=(9,5))

sns.barplot(x='RMSE',y='Train_Test',hue='Model',data=eval_metrics_df)

plt.xlabel('Train and Test Datsets')

plt.ylabel('RMSE Values')

plt.title('Model Comparison')

axes=plt.gca()

axes.set_xlim([0,4])