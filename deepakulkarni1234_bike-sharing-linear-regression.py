#Impoting all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import random
import warnings
warnings.filterwarnings('ignore')
#Reading the data from CSV file
df = pd.read_csv("F:/day.csv")
#Analysing the data
df.shape
#Analysing the data
df.info()
df.head()
#dropping unwanted columns
df=df.drop(['instant', 'dteday','temp','casual','registered'], axis=1)
#rechecking the data
df.head()
#Analysing the column Weathersit
df.weathersit.value_counts()
#Analysing the column Weekday
df.weekday.value_counts()
#Conversion of datatypes
df["weathersit"]= df["weathersit"].astype(object) 
df["season"]= df["season"].astype(object) 
df["mnth"]= df["mnth"].astype(object) 
df["weekday"]= df["weekday"].astype(object) 
#Renaming the values in Season column
df['season'] = np.where((df.season == 1) ,'spring',df.season)
df['season'] = np.where((df.season == 2) ,'summer',df.season)
df['season'] = np.where((df.season == 3) ,'fall',df.season)
df['season'] = np.where((df.season == 4) ,'winter',df.season)
#Renaming the values in weathersit column
df['weathersit'] = np.where((df.weathersit == 1) ,'Clear',df.weathersit)
df['weathersit'] = np.where((df.weathersit == 2) ,'Mist_Cloudy', df.weathersit)
df['weathersit'] = np.where((df.weathersit == 3) ,'Light_Snow_Light_Rain' ,df.weathersit)
df['weathersit'] = np.where((df.weathersit == 4) ,'Heavy Rain_Ice Pallets' ,df.weathersit)
#Renaming the values in weekday column
df['weekday'] = np.where((df.weekday == 0) ,'Sun',df.weekday)
df['weekday'] = np.where((df.weekday == 1) ,'Mon',df.weekday)
df['weekday'] = np.where((df.weekday == 2) ,'Tue',df.weekday)
df['weekday'] = np.where((df.weekday == 3) ,'Wed',df.weekday)
df['weekday'] = np.where((df.weekday == 4) ,'Thu',df.weekday)
df['weekday'] = np.where((df.weekday == 5) ,'Fri',df.weekday)
df['weekday'] = np.where((df.weekday == 6) ,'Sat',df.weekday)
#Re-Analysing the record in weathersit
#df.season.value_counts()
#df.mnth.value_counts()
df.weathersit.value_counts()
#Re-Analysing the record in season
df.season.value_counts()
#df.mnth.value_counts()
#df.weathersit.value_counts()
#Re-Analysing the record in weekday
df.weekday.value_counts()
# Count vs 'humidity', 'atemp', 'windspeed'(Continuous Values)
sns.pairplot(df, x_vars=['hum', 'atemp', 'windspeed'], y_vars='cnt',size=4, aspect=1, kind='scatter')
plt.show()
##Count vs'season','weekday','workingday','weathersit' ( Categorical Data)
sns.pairplot(df, x_vars=['season','weekday','workingday','weathersit'], y_vars='cnt',size=4, aspect=1, kind='scatter')
plt.show()
#Categotical Data Analaysis 
plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'weathersit', y = 'cnt', data = df)
plt.subplot(2,3,2)
sns.boxplot(x = 'season', y = 'cnt', data = df)
plt.subplot(2,3,3)
sns.boxplot(x = 'mnth', y = 'cnt', data = df)
plt.subplot(2,3,4)
sns.boxplot(x = 'weekday', y = 'cnt', data = df)
#Categotical Data Analaysis on Dependent Variables
plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'weathersit', y = 'atemp', data = df)
plt.subplot(2,3,2)
sns.boxplot(x = 'season', y = 'atemp', data = df)
plt.subplot(2,3,3)
sns.boxplot(x = 'mnth', y = 'atemp', data = df)
plt.subplot(2,3,4)
sns.boxplot(x = 'weekday', y = 'atemp', data = df)
plt.subplot(2,3,1)
plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'weathersit', y = 'windspeed', data = df)
plt.subplot(2,3,2)
sns.boxplot(x = 'season', y = 'windspeed', data = df)
plt.subplot(2,3,3)
sns.boxplot(x = 'mnth', y = 'windspeed', data = df)
plt.subplot(2,3,4)
sns.boxplot(x = 'weekday', y = 'windspeed', data = df)
#Count vs Season with Weekdays Analysis
plt.figure(figsize = (10, 5))
sns.boxplot(x = 'season', y = 'cnt', hue = 'weekday', data = df)
plt.show()
#Count vs Weahtersit and Weekdays
plt.figure(figsize = (20, 10))
sns.boxplot(x = 'weathersit', y = 'cnt', hue = 'weekday', data = df)
plt.show()
#Analysing the Holiday/Non-Holiday Count of Bike Renatals
Holiday_no=df.loc[df['holiday'] == 0, 'cnt'].sum()
Holiday_yes=df.loc[df['holiday'] == 1, 'cnt'].sum()
la = ['Holiday', 'Working'] 
  
data = [Holiday_yes, Holiday_no]
  
# Creating plot 
fig = plt.figure(figsize =(10, 7)) 
plt.pie(data, labels = la) 
  
# show plot 
plt.show() 
#correlation 
corrs = df[['atemp','windspeed','hum','cnt']].corr()
sns.heatmap(corrs,annot=True,vmax=3)
#Heat Map for the Factors 
plt.figure(figsize = (10, 10))
sns.heatmap(df.corr(), cmap="YlGnBu", annot = True)
plt.show()

#Dummifying Weather column 
weather = pd.get_dummies(df['weathersit'], drop_first = True)
df = pd.concat([df, weather], axis = 1)
weather .head()
#Dummifying Season column 
seasons = pd.get_dummies(df['season'], drop_first = True)
df = pd.concat([df, seasons], axis = 1)
seasons.head()
#Dummifying the Column month
month = pd.get_dummies(df['mnth'], drop_first = True)
df = pd.concat([df, month], axis = 1)
month.head()
#Dummifying month column 
weekdays = pd.get_dummies(df['weekday'], drop_first = True)
df = pd.concat([df, weekdays], axis = 1)
weekdays.head()
#Dropping the dummified columns from main Dataframe 
df.drop(['weathersit'], axis = 1, inplace = True)
df.drop(['season'], axis = 1, inplace = True)
df.drop(['mnth'], axis = 1, inplace = True)
df.drop(['weekday'], axis = 1, inplace = True)
df.head()
from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['atemp', 'hum', 'windspeed', 'cnt']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df_train.head()
plt.figure(figsize = (80, 80))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()

y_train = df_train.pop('cnt')
X_train = df_train
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
rfe=RFE(lm,15)
rfe=rfe.fit(X_train,y_train)

list(zip(X_train.columns,rfe.support_,rfe.ranking_))
#List of Columns selected after RFE 
col=X_train.columns[rfe.support_]
col
#Selection of those columns from X_train ( Train Data from DF )
X_train_rfe=X_train[col]
#adding Constant 
import statsmodels.api as sm 
X_train_rfe_1=sm.add_constant(X_train_rfe)
#fitting the data
lm_1=sm.OLS(y_train,X_train_rfe_1).fit()
#Printing the summary 
print(lm_1.summary())
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd 
#VIF Analysing the df Data frame 
#from statsmodels.stats.outlier_influence import variance_inflation_factor


Vif=pd.DataFrame()
Vif['Columm']=X_train_rfe.columns
Vif['VIF']=[variance_inflation_factor (X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
Vif['VIF']=round(Vif['VIF'],2)
Vif=Vif.sort_values(by='VIF',ascending=False)
Vif
X_train_rfe = X_train_rfe.drop('hum', 1,)
import statsmodels.api as sm 
X_train_rfe_2=sm.add_constant(X_train_rfe)

lm_2=sm.OLS(y_train,X_train_rfe_2).fit()
# Print the summary of the model
print(lm_2.summary())

Vif=pd.DataFrame()
Vif['Columm']=X_train_rfe.columns
Vif['VIF']=[variance_inflation_factor (X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
Vif['VIF']=round(Vif['VIF'],2)
Vif=Vif.sort_values(by='VIF',ascending=False)
Vif
X_train_rfe = X_train_rfe.drop('atemp', 1,)
import statsmodels.api as sm 
X_train_rfe_3=sm.add_constant(X_train_rfe)



lm_3=sm.OLS(y_train,X_train_rfe_3).fit()
# Print the summary of the model
print(lm_3.summary())

Vif=pd.DataFrame()
Vif['Columm']=X_train_rfe.columns
Vif['VIF']=[variance_inflation_factor (X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
Vif['VIF']=round(Vif['VIF'],2)
Vif=Vif.sort_values(by='VIF',ascending=False)
Vif
X_train_rfe = X_train_rfe.drop('winter', 1,)
import statsmodels.api as sm 
X_train_rfe_4=sm.add_constant(X_train_rfe)



lm_4=sm.OLS(y_train,X_train_rfe_4).fit()
# Print the summary of the model
print(lm_4.summary())

Vif=pd.DataFrame()
Vif['Columm']=X_train_rfe.columns
Vif['VIF']=[variance_inflation_factor (X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
Vif['VIF']=round(Vif['VIF'],2)
Vif=Vif.sort_values(by='VIF',ascending=False)
Vif
X_train_rfe = X_train_rfe.drop(4, 1,)
import statsmodels.api as sm 
X_train_rfe_5=sm.add_constant(X_train_rfe)



lm_5=sm.OLS(y_train,X_train_rfe_5).fit()
# Print the summary of the model
print(lm_5.summary())

Vif=pd.DataFrame()
Vif['Columm']=X_train_rfe.columns
Vif['VIF']=[variance_inflation_factor (X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
Vif['VIF']=round(Vif['VIF'],2)
Vif=Vif.sort_values(by='VIF',ascending=False)
Vif
X_train_rfe = X_train_rfe.drop('windspeed', 1,)
import statsmodels.api as sm 
X_train_rfe_6=sm.add_constant(X_train_rfe)



lm_6=sm.OLS(y_train,X_train_rfe_6).fit()
# Print the summary of the model
print(lm_6.summary())

Vif=pd.DataFrame()
Vif['Columm']=X_train_rfe.columns
Vif['VIF']=[variance_inflation_factor (X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
Vif['VIF']=round(Vif['VIF'],2)
Vif=Vif.sort_values(by='VIF',ascending=False)
Vif
#adding column 10 
X_train_rfe=X_train[['yr','spring','Mist_Cloudy',3, 8, 5, 9, 6, 10, 'holiday','Light_Snow_Light_Rain']]
X_train_rfe.head()
import statsmodels.api as sm 
X_train_rfe_7=sm.add_constant(X_train_rfe)



lm_7=sm.OLS(y_train,X_train_rfe_7).fit()
# Print the summary of the model
print(lm_7.summary())

Vif=pd.DataFrame()
Vif['Columm']=X_train_rfe.columns
Vif['VIF']=[variance_inflation_factor (X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
Vif['VIF']=round(Vif['VIF'],2)
Vif=Vif.sort_values(by='VIF',ascending=False)
Vif
X_train_rfe=X_train[['yr','spring',3, 8, 5, 9,6,'Mist_Cloudy','holiday','Light_Snow_Light_Rain','Sun',10]]
import statsmodels.api as sm 
X_train_rfe_8=sm.add_constant(X_train_rfe)



lm_8=sm.OLS(y_train,X_train_rfe_8).fit()
# Print the summary of the model
print(lm_8.summary())
Vif=pd.DataFrame()
Vif['Columm']=X_train_rfe.columns
Vif['VIF']=[variance_inflation_factor (X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
Vif['VIF']=round(Vif['VIF'],2)
Vif=Vif.sort_values(by='VIF',ascending=False)
Vif
#Taking into account Columns after RFE and adding further more Columns after previous Model steps

X_train_rfe=X_train[['yr','spring',3, 8, 5, 9,6,'Mist_Cloudy','holiday','Light_Snow_Light_Rain','Sun',10,7]]
#Fitting a model 
import statsmodels.api as sm 
X_train_rfe_9=sm.add_constant(X_train_rfe)

lm_9=sm.OLS(y_train,X_train_rfe_9).fit()
# Print the summary of the model
print(lm_9.summary())
#Vifs 
Vif=pd.DataFrame()
Vif['Columm']=X_train_rfe.columns
Vif['VIF']=[variance_inflation_factor (X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
Vif['VIF']=round(Vif['VIF'],2)
Vif=Vif.sort_values(by='VIF',ascending=False)
Vif

y_train_pred = lm_9.predict(X_train_rfe_9)
res = (y_train - y_train_pred)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_pred), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18) 
x_t=X_train_rfe_9.iloc[:,0].values
plt.scatter(x_t, res)
plt.show()
p = sns.scatterplot(y_train_pred,res)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
plt.ylim(-2,2)
plt.xlim(0,1)
p = sns.lineplot([0,26],[0,0],color='blue')
p = plt.title('Residuals vs fitted values plot for homoscedasticity check')
num_vars=['atemp','hum','windspeed','cnt']
df_test[num_vars]=scaler.fit_transform(df_test[num_vars])
y_test = df_test.pop('cnt')
X_test = df_test
X_test=X_test[['yr','spring',3, 8, 5, 9,6,'Mist_Cloudy','holiday','Light_Snow_Light_Rain','Sun',10,7]]
# Adding constant variable to test dataframe
X_test_sm = sm.add_constant(X_test)
y_pred = lm_9.predict(X_test_sm)
fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)      
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#Returns the mean squared error; we'll take a square root
np.sqrt(mean_squared_error(y_test, y_pred))
#R square for Test data on built model 
r_squared = r2_score(y_test, y_pred)
r_squared
# Agjusted R square for Test data on built model 
print("Rsquared_adj\n",lm_9.rsquared_adj)
