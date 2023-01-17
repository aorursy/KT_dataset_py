import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error
df = pd.read_csv('../input/boombikes/day.csv')

df.head()
# Checking it's Shape

df.shape
# Checking if instant column has unique entries, if yes, then will convert it to index

df['instant'].nunique()
# Setting the instant column as index to number of columns

df.set_index('instant', inplace=True)

df.head()
df.drop(['casual', 'registered'], inplace=True, axis = 1)

df.head()
# Checking Columns Data Types

df.info()
# Converting the dtedat to Date Time

df['dteday'] = pd.to_datetime(df['dteday'])

df['dteday'].dtypes
# Changing the month number to month abbr for better view

import calendar

df['mnth'] = df['mnth'].apply(lambda x: calendar.month_abbr[x])

df['mnth'].unique()
# Since season, weekday and weathesit are basically categorical values, converting them to string type for future use

df[['season','weekday','weathersit']] = df[['season','weekday','weathersit']].astype(str)
df.info()
# Checking the % of null values in each column

round(df.isnull().sum()/len(df.index)*100,2)
# Checking the range of values, for example temperature, humidity etc

df[['temp', 'atemp', 'hum', 'windspeed']].describe()
# Converting the weather variable into more understanable text

df['weathersit'].replace(['1','2','3','4'],['Good', 'Average', 'Bad', 'Very Bad'], inplace=True)
# Converting the seasons into specific season names for better understanding

df['season'].replace(['1','2','3','4'],['spring', 'summer', 'fall', 'winter'], inplace=True)
# Checking linear relationship between the cnt variable and other numeric variables

x =sns.pairplot(df, palette='husl', x_vars=['temp', 'atemp', 'hum', 'windspeed'], y_vars=['cnt'] , hue='yr' )

x._legend.remove()

plt.legend(labels=['2018', '2019'])

plt.show()
# Checking the distribution of rentals across different categorical variables

plt.figure(figsize=(15,10))

plt.subplot(2,3,1)

sns.boxplot(x='season', y='cnt', data=df, palette='husl')

plt.subplot(2,3,2)

sns.boxplot(x='yr', y='cnt', data=df, palette='husl')

plt.subplot(2,3,3)

sns.boxplot(x='mnth', y='cnt', data=df, palette='husl')

plt.subplot(2,3,4)

sns.boxplot(x='holiday', y='cnt', data=df, palette='husl')

plt.subplot(2,3,5)

sns.boxplot(x='weekday', y='cnt', data=df, palette='husl')

plt.subplot(2,3,6)

sns.boxplot(x='workingday', y='cnt', data=df, palette='husl')



plt.show()
sns.boxplot(x='weathersit', y='cnt', data=df, palette='husl')

plt.xlabel('Weather')

plt.show()
# Checking business on Holidays

holiday_df = df.groupby(['holiday'])['cnt'].mean().reset_index()

sns.barplot(x='holiday', y='cnt', data=holiday_df, palette='husl')

plt.xticks(np.arange(2),('No','Yes'))

plt.xlabel('Holiday')

plt.ylabel('Average Number of Rentals')

plt.show()
# Total rentals on different days of the week.

weekday_df = df.groupby(['weekday'])['cnt'].mean().reset_index()

sns.barplot(x='weekday', y='cnt', data=weekday_df, palette='husl')

plt.xticks(np.arange(7),('Mon','Tue','Wed','Thu', 'Fri', 'Sat', 'Sun'))

plt.xlabel('Days of the Week')

plt.ylabel('Average Number of Rentals')

plt.show()
# Checking business on Workingdays

workingday_df = df.groupby(['workingday'])['cnt'].mean().reset_index()

sns.barplot(x='workingday', y='cnt', data=workingday_df, palette='husl')

plt.xticks(np.arange(2),('No','Yes'))

plt.xlabel('Working Day')

plt.ylabel('Average Number of Rentals')

plt.show()
dummy = pd.get_dummies(df[['season','mnth','weekday','weathersit']], drop_first=True)

dummy.head()
df = pd.concat([df,dummy], axis=1)   #Axis=1 is for horizontal stacking

df = df.drop(['season','mnth','weekday','weathersit'], axis=1)

df.head()
print('Shape of the new dataframe is:' , df.shape)
# Since we have the month and the Year in two seperate columns, we do not need the date column anymore, thus dropping it

df.drop('dteday', inplace=True, axis = 1)
# Moving the cnt to the end for easier identification

first_col = df.pop('cnt')

df['cnt'] = first_col
df_train, df_test = train_test_split(df, train_size=0.7, random_state=100)
print('Shape of the Train data is:' , df_train.shape)

print('Shape of the Test data is:' , df_test.shape)
# Checking the Train Data

pd.set_option('display.max_columns', None)

df_train.head()
# We do a MinMax scaling

scaler = MinMaxScaler()    #Instantiating the object

cols = df_train.columns

df_train[cols] = scaler.fit_transform(df_train[cols])
# Checking the Heatmap

plt.figure(figsize=(24,15))

sns.heatmap(df_train.corr(),annot=True, cmap='YlGnBu')

plt.show()
y_train = df_train.pop('cnt')

X_train = df_train

X_train_sm = sm.add_constant(X_train)

lr = sm.OLS(y_train, X_train_sm)

lr_model1 = lr.fit()

lr_model1.summary()
# Checking VIF (Variance Inflation Factor - MultiColinearity)

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Removing 'mnth_Mar' due to high P-Value

X = X_train.drop('mnth_Mar',axis=1)

X_train_sm = sm.add_constant(X)

lr = sm.OLS(y_train, X_train_sm)

lr_model2 = lr.fit()

lr_model2.summary()
# Removing 'weekday_4' due to high P-Value

X = X.drop('weekday_4',axis=1)

X_train_sm = sm.add_constant(X)

lr = sm.OLS(y_train, X_train_sm)

lr_model3 = lr.fit()

lr_model3.summary()
# Removing 'mnth_Oct' due to high P-Value

X = X.drop('mnth_Oct',axis=1)

X_train_sm = sm.add_constant(X)

lr = sm.OLS(y_train, X_train_sm)

lr_model4 = lr.fit()

lr_model4.summary()
# Removing 'mnth_Jun' due to high P-Value

X = X.drop('mnth_Jun',axis=1)

X_train_sm = sm.add_constant(X)

lr = sm.OLS(y_train, X_train_sm)

lr_model5 = lr.fit()

lr_model5.summary()
# Removing 'weekday_3' due to high P-Value

X = X.drop('weekday_3',axis=1)

X_train_sm = sm.add_constant(X)

lr = sm.OLS(y_train, X_train_sm)

lr_model6 = lr.fit()

lr_model6.summary()
# Removing 'atemp' due to high P-Value

X = X.drop('atemp',axis=1)

X_train_sm = sm.add_constant(X)

lr = sm.OLS(y_train, X_train_sm)

lr_model7 = lr.fit()

lr_model7.summary()
# Removing 'weekday_5' due to high P-Value

X = X.drop('weekday_5',axis=1)

X_train_sm = sm.add_constant(X)

lr = sm.OLS(y_train, X_train_sm)

lr_model8 = lr.fit()

lr_model8.summary()
# Removing 'mnth_Aug' due to high P-Value

X = X.drop('mnth_Aug',axis=1)

X_train_sm = sm.add_constant(X)

lr = sm.OLS(y_train, X_train_sm)

lr_model9 = lr.fit()

lr_model9.summary()
# Removing 'weekday_2' due to high P-Value

X = X.drop('weekday_2',axis=1)

X_train_sm = sm.add_constant(X)

lr = sm.OLS(y_train, X_train_sm)

lr_model10 = lr.fit()

lr_model10.summary()
# Removing 'weekday_1' due to high P-Value

X = X.drop('weekday_1',axis=1)

X_train_sm = sm.add_constant(X)

lr = sm.OLS(y_train, X_train_sm)

lr_model11 = lr.fit()

lr_model11.summary()
# Removing 'mnth_May' due to high P-Value

X = X.drop('mnth_May',axis=1)

X_train_sm = sm.add_constant(X)

lr = sm.OLS(y_train, X_train_sm)

lr_model12 = lr.fit()

lr_model12.summary()
# Removing 'mnth_Feb' due to high P-Value

X = X.drop('mnth_Feb',axis=1)

X_train_sm = sm.add_constant(X)

lr = sm.OLS(y_train, X_train_sm)

lr_model13 = lr.fit()

lr_model13.summary()
# Checking VIF (Variance Inflation Factor - MultiColinearity)

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Removing 'hum' due to high VIF

X = X.drop('hum',axis=1)

X_train_sm = sm.add_constant(X)

lr = sm.OLS(y_train, X_train_sm)

lr_model14 = lr.fit()

lr_model14.summary()
#Checking the VIF Again

vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Checking the co-efficients of the final model lr_model14

print(lr_model14.summary())
# Validating Linear Relationship

sm.graphics.plot_ccpr(lr_model14, 'temp')

plt.show()
# Validating Homoscedasticity : The residuals have constant variance with respect to the dependent variable

y_train_pred = lr_model14.predict(X_train_sm)

sns.scatterplot(y_train,(y_train - y_train_pred))

plt.plot(y_train,(y_train - y_train), '-r')

plt.xlabel('Count')

plt.ylabel('Residual')

plt.show()
# Validating Multi Colinearity

plt.figure(figsize=(15,8))

sns.heatmap(X.corr(),annot=True, cmap='YlGnBu')

plt.show()
print(vif)
# Independence of residuals (absence of auto-correlation)

# Autocorrelation refers to the fact that observations’ errors are correlated

# To verify that the observations are not auto-correlated, we can use the Durbin-Watson test. 

# The test will output values between 0 and 4. The closer it is to 2, the less auto-correlation there is between the various variables

# (0–2: positive auto-correlation, 2–4: negative auto-correlation)



print('The Durbin-Watson value for Model No.14 is',round(sm.stats.stattools.durbin_watson((y_train - y_train_pred)),4))
# Normality of Errors

y_train_pred = lr_model14.predict(X_train_sm)



# Ploting the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_pred))

fig.suptitle('Error Terms')                  

plt.xlabel('Errors')     

plt.show()
sm.qqplot((y_train - y_train_pred), fit=True, line='45')

plt.show()
# Scaling the Test Dataset with the Scaler of the Training Set

cols = df_test.columns

df_test[cols] = scaler.transform(df_test[cols])
# Dividing into X_test and y_test

y_test = df_test.pop('cnt')

X_test = df_test
# Adding the constant column

X_test_m14 = sm.add_constant(X_test)

# Removing all the columns which has been removed from Model 14

X_test_m14 = X_test_m14.drop(['hum','mnth_Feb','mnth_Mar','mnth_May',

                              'mnth_Jun','mnth_Aug','mnth_Oct','atemp',

                              'weekday_1','weekday_2','weekday_3','weekday_4','weekday_5' ], axis=1)
# Making prediction using Model 14

y_test_pred = lr_model14.predict(X_test_m14)
print('The R-Squared score of the model for the predicted values is',round(r2_score(y_test, y_test_pred),2))

print('The Root Mean Squared Error of the model for the predicted values is',round(np.sqrt(mean_squared_error(y_test, y_test_pred)),4))

print('The Mean Absolute Error of the model for the predicted values is',mean_absolute_error(y_test, y_test_pred))
# As asked in problem statement

from sklearn.metrics import r2_score

r2_score(y_test, y_test_pred)