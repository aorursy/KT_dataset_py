import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set()
# Read the dataset



bike= pd.read_csv("../input/bike-sharing-demand-day/day.csv")
# Let's see how our dataset looks like



bike.head()
# Let's see how many rows and columns we have



bike.shape
#Let's get some information on the dataset



bike.info()
# Let's see some summary



bike.describe()
# To check if there are any missing values in the dataset



import missingno as mn

mn.matrix(bike)
bike['dteday'].dtype
bike['dteday'] =  pd.to_datetime(bike['dteday'],format='%d-%m-%Y')

bike['dteday'].dtype
bike['year'] = pd.DatetimeIndex(bike['dteday']).year

bike['month'] = pd.DatetimeIndex(bike['dteday']).month
bike.head()
# Dropping the columns as we have extracte#d the correct year and month from the date.



bike.drop(['yr','mnth'],axis=1,inplace=True)
bike.head()
#Dropping the redundant variable holiday as the workingday column covers enough information that is required.



bike.drop('holiday',axis=1,inplace=True)
# Dropping the dteday,instant,casual and registered columns.



bike.drop(['dteday','instant','casual','registered'],axis=1,inplace=True)
bike.head()
# Renaming some columns for better understanding



bike.rename(columns={'hum':'humidity','cnt':'count'},inplace=True)
bike.head()
codes = {1:'spring',2:'summer',3:'fall',4:'winter'}

bike['season'] = bike['season'].map(codes)
sns.barplot('season','count',data=bike)
codes = {1:'Clear',2:'Mist',3:'Light Snow',4:'Heavy Rain'}

bike['weathersit'] = bike['weathersit'].map(codes)
sns.barplot('weathersit','count',data=bike)
codes = {1:'working_day',0:'Holiday'}

bike['workingday'] = bike['workingday'].map(codes)
sns.barplot('workingday','count',data=bike,palette='cool')
codes = {2019:1,2018:0}

bike['year'] = bike['year'].map(codes)
sns.barplot('year','count',data=bike,palette='dark')
codes = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'June',7:'July',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

bike['month'] = bike['month'].map(codes)
plt.figure(figsize=(10,5))

sns.barplot('month','count',hue='year',data=bike,palette='Paired')
codes = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

bike['weekday'] = bike['weekday'].map(codes)
bike.groupby('weekday')['count'].max().plot(kind='bar')
plt.scatter('temp','count',data=bike)
plt.scatter('atemp','count',data=bike)
plt.scatter('humidity','count',data=bike)
plt.scatter('windspeed','count',data=bike)
sns.distplot(bike['count'])
sns.pairplot(bike)
plt.figure(figsize = (12,6))

sns.heatmap(bike.corr(),annot=True)
data= bike[['temp','atemp','humidity','windspeed']]

sns.heatmap(data.corr(),annot=True)
bike.drop('atemp',axis=1,inplace=True)
bike.head()
seasons = pd.get_dummies(bike['season'],drop_first=True)



working_day = pd.get_dummies(bike['workingday'],drop_first=True)



weather= pd.get_dummies(bike['weathersit'],drop_first=True)



month= pd.get_dummies(bike['month'],drop_first=True)



week_day= pd.get_dummies(bike['weekday'],drop_first=True)
bike= pd.concat([bike,seasons,working_day,weather,month,week_day],axis=1)
bike.head()
# Dropping the categorical variables as they are already dummy-encoded.



bike.drop(['season','workingday','weathersit','weekday','month'],axis=1,inplace=True)
bike.head()
from sklearn.model_selection import train_test_split



np.random.seed(0)

df_train, df_test = train_test_split(bike, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
# Apply scaler() to all the columns except the'dummy' variables.



num_vars=['temp','humidity','windspeed','count']



df_train[num_vars]= scaler.fit_transform(df_train[num_vars])
plt.scatter('temp','count',data=df_train)
y_train = df_train.pop('count')

X_train = df_train
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 10



lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm,10) # running RFE

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]

col
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model
lm.summary()
X_train1= X_train_rfe.drop('Mon',1)
X_train2= sm.add_constant(X_train1)

lm1 = sm.OLS(y_train,X_train2).fit() 
lm1.summary()
X_train_new= X_train2.drop('const',axis=1)
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_pred = lm1.predict(X_train2)
fig = plt.figure()

sns.distplot((y_train - y_train_pred), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)     
num_vars=['temp','humidity','windspeed','count']



df_test[num_vars]= scaler.transform(df_test[num_vars])
y_test = df_test.pop('count')

X_test = df_test
# Now let's use our model to make predictions.



# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_new.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_test_pred = lm1.predict(X_test_new)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_test_pred)

fig.suptitle('Actual vs Predictions', fontsize=20)              # Plot heading 

plt.xlabel('Actual', fontsize=18)                          # X-label

plt.ylabel('Predictions', fontsize=16)                          # Y-label
from sklearn.metrics import r2_score

r2_score(y_test, y_test_pred)