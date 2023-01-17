#importing required libraries and packages

import numpy as np

np.random.seed(0)

import pandas as pd

import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor



from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import mean_squared_error as mse



pd.set_option('max_columns', 100)
#Function Definitions



#Function to get the VIFs for all the variables in a dataframe

def getvif(df):

    if 'const' in list(df.columns):

        df1=df.drop('const', axis=1) 

    else:

        df1 = df.copy()

    vif=pd.DataFrame()

    vif['Features'] = df1.columns

    vif['VIF'] = [variance_inflation_factor(df1.values, i) for i in range(df1.shape[1])]

    vif['VIF'] = round(vif.VIF,2)

    vif = vif.sort_values(by = 'VIF', ascending = False)

    return vif





#Function to print Model Evaluation parameters

def modeleval(y_actual, y_pred, x_test):

    print('R2 Score of Model')

    print(r2_score(y_true = y_actual, y_pred = y_pred))

    print('\nMean Absolute Error')

    print(mae(y_true = y_actual, y_pred = y_pred))

    print('\nMean Squared Error')

    print(mse(y_true = y_actual, y_pred = y_pred))

    print('\nRoot Mean Squared Error')

    print(np.sqrt(mse(y_true = y_actual, y_pred = y_pred)))

    print('\nAdjusted R-squared')

    print(1 - (1-r2_score(y_true = y_actual, y_pred = y_pred))*(len(y_actual)-1)/(len(y_actual)-x_test_m6.shape[1]-1))
#importing the data

df = pd.read_csv('../input/bike-sharing-demand/day.csv')

df.head()
df.shape
df.info()
df.describe()
#There are no null values

df.isnull().sum()
#Dropping columns we would not be needing/using

#Instant is just an identifier

#dteday is the date and can be analyzed using other columns in the dataset, so it is redundant

#casual and registered are part of cnt

df.drop(['instant','dteday','casual','registered'], axis = 1, inplace = True)
#Getting a pairplot for the entire dataset

pp = sns.pairplot(df)

fig = pp.fig

fig.savefig("output.png")
#Visalizing correlation between different varaibles

fig, ax = plt.subplots(figsize=(18,18))

sns.heatmap(df.corr(), annot=True, ax=ax)

plt.show()
#We see a high correlation between temp and atemp. Both of those together will not be crucial to the model and will lead to a very high (maybe infinite) VIF. 

#Hence, dropping atemp

df.drop(['atemp'], axis = 1, inplace = True)
#Visalizing categorical variables

plt.figure(figsize = (20,20))

catcols = ['season','yr','holiday','weekday','workingday','weathersit']

for i in range(1,7):

    plt.subplot(2,3,i)

    sns.boxplot(x = catcols[i-1], y = 'cnt', data = df)



plt.savefig('box')    
#Let's see the AVERAGE ridership across categorical variables

plt.figure(figsize=(16,16))

plt.subplot(2,3,i)

for i in range(1,7):

    plt.subplot(2,3,i)

    df.groupby(catcols[i-1])['cnt'].mean().plot.barh()

    plt.xlabel('Average Ridership')

plt.savefig('barh.png')

plt.show()

#Let's see the TOTAL ridership across variables

plt.figure(figsize=(16,16))

plt.subplot(2,3,i)

for i in range(1,7):

    plt.subplot(2,3,i)

    df.groupby(catcols[i-1])['cnt'].sum().plot.barh()

plt.show()
df.head()
#changing the integer encodings to string for the season, mnth, weekday and creating dummy variables for all of these (since they do not have an associated cardinality)

df.season = df.season.map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})

seasons = pd.get_dummies(df.season, drop_first = True)

df.mnth = df.mnth.map({1:'jan', 2:'feb', 3:'mar', 4:'apr',5:'may', 6:'jun', 7:'jul', 8:'aug',9:'sep', 10:'oct', 11:'nov', 12:'dec'})

months = pd.get_dummies(df.mnth, drop_first = True)

df.weekday = df.weekday.map({1:'mon',2:'tue',3:'wed',4:'thu',5:'fri',6:'sat',0:'sun'})

weekdays = pd.get_dummies(df.weekday, drop_first = True)

df.weathersit = df.weathersit.map({1:'clear',2:'cloudy',3:'light rain',4:'heavy rain'})

weather = pd.get_dummies(df.weathersit, drop_first = True)
#Inspecting the created dummy variables (already dropped the first one)
seasons.head()
months.head()
weekdays.head()
weather.head()
#Let's add these dummy variables to the data frame and remove the original columns which are now dummified

df = pd.concat([df,seasons,months,weekdays,weather], axis = 1)

df.drop(['season','mnth','weekday','weathersit'],axis = 1, inplace = True)

df.head()
#Splitting the data in 70:30 ratio

df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)
df_train.shape
df_test.shape
#Now that we have split the data, we can work on scaling the variables in the train data set.

#We will be using the MinMaxScaler

scaler = MinMaxScaler()

to_scale = ['temp', 'hum', 'windspeed','cnt']

#fitting the scaler on the training dataset only

df_train[to_scale] = scaler.fit_transform(df_train[to_scale])
df_train.head()
df_train.describe()
#Visualizing correlation between all the variables now in the data

fig, ax = plt.subplots(figsize=(18,18))

sns.heatmap(df_train.corr(), annot = True, ax = ax)

plt.show()
#defining x_train and y_train, x_test and y_test

x_train = df_train.drop(['cnt'], axis=1)

y_train = df_train[['cnt']]



x_test = df_test.drop(['cnt'], axis = 1)

y_test = df_test[['cnt']]
#Perform Recursive Feature Elimination 

lm = LinearRegression()

lm.fit(x_train, y_train)



rfe = RFE(lm, 15) #We will set number of output variables to 15

rfe = rfe.fit(x_train , y_train)

list(zip(x_train.columns, rfe.support_ , rfe.ranking_ ))
#Let's see which columns were selected

cols_selected = x_train.columns[rfe.support_]

cols_selected
#And these columns would be eliminated

x_train.columns[~rfe.support_]
#Creating another dataframe with only the retained variables from x_train

x_train_rfe = x_train[cols_selected]
#Let's see the correlation heatmap once again for these variables

fig, ax = plt.subplots(figsize=(18,18))

sns.heatmap(x_train_rfe.corr(), annot = True, ax = ax)

plt.show()
#Adding a constant term since sklearn does not automatically add constant to the model

x_train_rfe = sm.add_constant(x_train_rfe)
#Running the linear model on the now ready data

lm = sm.OLS(y_train, x_train_rfe).fit()

print(lm.summary())
#Checking VIF

getvif(x_train_rfe)
#Dropping hum since it has a very high VIF and most of the impact from this variable can be explained by other variables 

x_train_rfe2 = x_train_rfe.drop('hum',axis=1)
#Building new model without hum variable

lm2 = sm.OLS(y_train, x_train_rfe2).fit()

print(lm2.summary())
#Checking VIF for variables in new model

getvif(x_train_rfe2)
#Dropping holiday

x_train_rfe3 = x_train_rfe2.drop('holiday', axis = 1)
lm3 = sm.OLS(y_train, x_train_rfe3).fit()

print(lm3.summary())
#Checking VIF again 

getvif(x_train_rfe3)
#Dropping variable jan since it is turning out to be realtively less significant

x_train_rfe4 = x_train_rfe3.drop('jan', axis = 1)
lm4 = sm.OLS(y_train, x_train_rfe4).fit()

print(lm4.summary())
#checking VIFs

getvif(x_train_rfe4)
#Dropping hul since it realtively less significant

x_train_rfe5 = x_train_rfe4.drop('jul', axis = 1)
lm5 = sm.OLS(y_train, x_train_rfe5).fit()

print(lm5.summary())
#Checking VIFs

getvif(x_train_rfe5)
#Removing spring variable due to a higher VIF and lower significance

x_train_rfe6 = x_train_rfe5.drop('spring', axis=1)
lm6 = sm.OLS(y_train, x_train_rfe6).fit()

print(lm6.summary())
getvif(x_train_rfe6)
#Let's see the correlation heatmap once again for these variables

fig, ax = plt.subplots(figsize=(18,18))

sns.heatmap(x_train_rfe6.corr(), annot = True, ax = ax)

plt.show()
#predicting y_train using the model 7 

y_train_pred = lm6.predict(x_train_rfe6)
# Plotting y_train and y_train_pred to understand the variance from actual results in the train data

#This verifies the assumption of linearity of variables x and y (and hence linearity of the model)



fig = plt.figure()

sns.regplot(x=y_train, y=y_train_pred)

fig.suptitle('y_train vs y_pred')             

plt.xlabel('y_train')                          

plt.ylabel('y_pred')     
#Lets see the distribution of error terms in the training set

#it is coming out as normal

plt.figure(figsize=(8,8))

sns.distplot((y_train - y_train_pred.values.reshape(-1,1)))

plt.xlabel('Residuals')

plt.show()
#Checking residuals with a qqplot, the points are either on or close to the 45 degree line, indicating that the distribution is normal

sm.qqplot(lm6.resid, line='45',fit=True)
#checking multicollinearity in the final model using test data.

#The max value of VIF is 4.76, which is acceptable.

#This indicates there is very little multicollinearity between our selected variables and the assumption is met.

getvif(x_train_rfe6)
#Mean of residuals is also very close to zero

(y_train - y_train_pred.values.reshape(-1,1)).mean()
#We can also plot a regression line through our data to get a clear picture of the spread

#this helps us understand that error terms are independent of each other and mostly have a constant variance

sns.regplot(x=y_train_pred, y = (y_train - y_train_pred.values.reshape(-1,1))['cnt'])

plt.title('Spread of Residuals in Train Data')

plt.xlabel('Target Variable Prediction')

plt.ylabel('Residual Value')

plt.show()
#Visualzing residual terms

res_train = y_train - y_train_pred.values.reshape(-1,1)

sns.scatterplot(x=res_train.index, y=res_train.values.reshape(-1,))

plt.show()
#We will only use the transform method and not fit_transform since fitting is done on the train set only

df_test[to_scale] = scaler.transform(df_test[to_scale])
df_test.describe()
### Dividing into x_test and y_test

y_test = df_test.pop('cnt')

x_test = df_test
y_test.head()
x_test.head()
#adding constant for using sm

x_test_sm = sm.add_constant(x_test)
#Creating a predictor dataframe by only retaining the variables we retained in our model (Model 6)

x_test_m6 = x_test_sm[(x_train_rfe6.columns)]
x_test_m6.head()
#Making predictions using Model 6

y_test_pred_m6 = lm6.predict(x_test_m6)
#Performance of model 6 on test data

modeleval(y_test,y_test_pred_m6, x_test_m6)
# Plotting y_test and y_pred to understand the variance from actual results in the test data

#This verifies the assumption of linearity of variables x and y (and hence linearity of the model)



fig = plt.figure()

sns.regplot(x=y_test, y=y_test_pred_m6)

fig.suptitle('y_test vs y_pred')             

plt.xlabel('y_test')                          

plt.ylabel('y_pred')      
#Error terms are approximately normally distributed in the test data as well, thus meeting the assumption of linear regression

sns.distplot(((y_test - y_test_pred_m6)))

plt.xlabel('Residuals')

plt.show()
#Mean of residuals is also very close to zero

(y_test - y_test_pred_m6).mean()
#checking multicollinearity in the final model using test data.

#The max value of VIF is 4.99, which is acceptable.

#This indicates there is very little multicollinearity between our selected variables and the assumption is met.

getvif(x_test_m6)
#We can also plot a regression line through this data to get a clear picture of the spread

#this helps us understand that error terms are independent of each other and mostly have a constant variance

sns.regplot(x=y_test_pred_m6, y = y_test - y_test_pred_m6)

plt.title('Spread of Residuals in Test Data')

plt.xlabel('Target Variable Prediction')

plt.ylabel('Residual Value')

plt.show()
#Parameters of the final model

lm6.params.sort_values()