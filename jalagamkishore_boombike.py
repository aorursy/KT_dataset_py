# surpress warnings
import warnings
warnings.filterwarnings('ignore')

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# importing day.csv and looking at five columns
boombike = pd.read_csv("../input/daybmb/day.csv")
boombike.head()
boombike.shape
boombike.info()
# droping casual resistered as cnt = casual + registered and dteday as all other variables like 
# season weekday are derived from it
boombike = boombike.drop(['casual','registered','instant','dteday'],axis=1)
boombike.head()
# converting categorical variables datatype feom int to object
boombike['mnth'] = boombike['mnth'].astype(str)
boombike['weekday'] = boombike['weekday'].astype(str)
boombike['weathersit'] = boombike['weathersit'].astype(str)
boombike['season'] = boombike['season'].astype(str)
boombike.info()
# replcaing integers with month in mnth column
boombike['mnth'] =boombike['mnth'].replace('1','jan').replace('2','feb').replace('3','mar').replace('4','apr').replace('5',
'may').replace('6','jun').replace('7','july').replace('8','aug').replace('9','sep').replace('10','oct').replace('11',
 'nov').replace('12','dec')


# replcaing integers with season in season column
boombike['season'] =boombike['season'].replace('1','spring').replace('2','summer').replace('3','fall').replace('4','winter')

# replcaing integers with weekday in weekday column
boombike['weekday'] =boombike['weekday'].replace('1','mon').replace('2','tue').replace('3','wed').replace('4','thu').replace('5',
'fri').replace('6','sat').replace('0','sun')
# replcaing integers with weathersit in weathersit column
boombike['weathersit'] =boombike['weathersit'].replace('1','clear').replace('2','mist').replace('3',
'lightsnow').replace('4','rain')

boombike.head()
# pairplot between the numerical variables
sns.pairplot(boombike,vars=['temp','atemp','hum','windspeed','cnt'])
plt.show()
# box plot between the cnt and other categorical variables
plt.figure(figsize=(20,12))
plt.subplot(3,3,1)
sns.boxplot(x='season', y='cnt', data = boombike)
plt.subplot(3,3,2)
sns.boxplot(x='yr', y='cnt', data = boombike)
plt.subplot(3,3,3)
sns.boxplot(x='holiday', y='cnt', data = boombike)
plt.subplot(3,3,4)
sns.boxplot(x='weekday', y='cnt', data = boombike)
plt.subplot(3,3,5)
sns.boxplot(x='workingday', y='cnt', data = boombike)
plt.subplot(3,3,6)
sns.boxplot(x='mnth', y='cnt', data = boombike)
plt.subplot(3,3,7)
sns.boxplot(x='weathersit',y= 'cnt' ,data = boombike)
# creating dummy variables for season, weekday, mnth and weathersit 
season_1 = pd.get_dummies(boombike['season'],drop_first='True')
weekday_1 = pd.get_dummies(boombike['weekday'],drop_first='True')
mnth_1 = pd.get_dummies(boombike['mnth'],drop_first='True')
weathersit_1 = pd.get_dummies(boombike['weathersit'],drop_first='True')
#five columns in weathersit_1
weathersit_1.head() 
# five columns in mnth
mnth_1.head() 
#five columns in weekday
weekday_1.head()
# five columns in season
season_1.head()
# adding results to our original data frame boombike
boombike = pd.concat([boombike,season_1,weekday_1,mnth_1,weathersit_1], axis=1)
boombike.head()
# dropping season,weekday,mnth,weathersit columns as we have created dummies for it
boombike = boombike.drop(['season','weekday','mnth','weathersit'],axis=1)
boombike.head()
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(boombike, train_size = 0.7, test_size = 0.3, random_state = 100)
# using minmax scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_vars = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

df_train.head()
# heat map for seeing the corelation between the variables
plt.figure(figsize = (30, 20))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()
y_train = df_train.pop('cnt')
X_train = df_train
# importing RFE and linearregression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
#running rfe with outnumber of variables equal to 15
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 15)             
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]
col
X_train.columns[~rfe.support_]
# creating X_train dataframe with rfe selected variables
X_train_rfe = X_train[col]
import statsmodels.api as sm
X_train_rfe = sm.add_constant(X_train_rfe)
lr = sm.OLS(y_train,X_train_rfe)
lr_model =lr.fit()
lr_model.summary()
# all p values are significant.lets check VIF
X_train_rfe.columns
X_train_new = X_train_rfe.drop('const',axis=1)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# dropping hum column as as it is having highest vif valu and greater than 5
X_train_new = X_train_new.drop('hum',axis=1)
X_train_lm = sm.add_constant(X_train_new)
lr = sm.OLS(y_train,X_train_lm)
lr_model =lr.fit()
lr_model.summary()
# all the p values are significant and lets check for vif
X_train_new = X_train_lm.drop('const',axis=1)
vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# though temp is having high vif, bt dropping it Rsquared value dropping to .66 which is bad lets drop next highest vif working day
X_train_new = X_train_new.drop('workingday',axis=1)
X_train_lm = sm.add_constant(X_train_new)
lr = sm.OLS(y_train,X_train_lm)
lr_model =lr.fit()
lr_model.summary()
# dropping sat as pvalue > .05
X_train_new = X_train_new.drop('sat',axis=1)
X_train_lm = sm.add_constant(X_train_new)
lr = sm.OLS(y_train,X_train_lm)
lr_model =lr.fit()
lr_model.summary()
# all pvalues are significant lets check vif
X_train_new = X_train_lm.drop('const',axis=1)
vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# dropping windspeed as it is having high vif next to temp
X_train_new = X_train_new.drop('windspeed',axis=1)
X_train_lm = sm.add_constant(X_train_new)
lr = sm.OLS(y_train,X_train_lm)
lr_model =lr.fit()
lr_model.summary()
# dropping jan column as p >.05
X_train_new = X_train_new.drop('jan',axis=1)
X_train_lm = sm.add_constant(X_train_new)
lr = sm.OLS(y_train,X_train_lm)
lr_model =lr.fit()
lr_model.summary()
# all p values are significant lets check vif
X_train_new = X_train_lm.drop('const',axis=1)
vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# all vif values are below 5 and our Rsquared value is .824 and adjusted R square is .821 which is good
# checking whether all the error values are normally distributed ot not. lets plot histogram of error values


y_train_cnt = lr_model.predict(X_train_lm)
# Importing the required libraries for plots.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_cnt), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label
# applying the test scaling on X_test and y_test
num_vars =['temp','atemp','hum','windspeed','cnt']
df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.head()
#creating y_test and X_test
y_test =df_test.pop('cnt')
X_test = df_test
X_test.head()
# creating X_test_now by dropping columns which are dropped in training set
X_test_new = X_test[X_train_new.columns]
# adding constant
X_test_new = sm.add_constant(X_test_new)
# making predictions
y_test_pred = lr_model.predict(X_test_new)
# checking r2_score to know about our model whether it is good or bad
# importing libary
from sklearn.metrics import r2_score
r2_score(y_true =y_test,y_pred=y_test_pred)
# got r2_score .80 which is close to adjusted Rsuared value.82  tells that our model is good
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_test_pred)
fig.suptitle('y_test vs y_test_pred', fontsize=20)         # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label

# from abive we can see the spread of y_test vs Y-test_pred is good
# for plotting bar plot vs y_tes vs y_test_predict
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

df1 = df.head(50)
df1
# barplot for Actual vs predicted
df1.plot(kind='bar',figsize=(50,40))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
# from above bar plot we can we actual & precicted values are close which tells us that our model is good
