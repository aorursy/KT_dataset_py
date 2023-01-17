# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings 

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import PowerTransformer

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import RFE

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

import statsmodels.api as sm

from pandas.plotting import autocorrelation_plot

from statsmodels.stats.outliers_influence import variance_inflation_factor

from yellowbrick.regressor import ResidualsPlot

from statsmodels.graphics import tsaplots





pd.set_option('max_rows',None)

pd.set_option('display.max_columns',100)



%matplotlib inline
# Loading the data



path = r'/kaggle/input/bike-sharing-using-linear-regression/day.csv'



bike = pd.read_csv(path,parse_dates=['dteday'])



bike.head()
bike.shape
bike.info()
bike.describe()
# Renaming the columns



bike.rename(columns = {'hum':'Humidity','yr':'Year','temp':'Temperature','weathersit':'Weather','holiday':'Holiday','cnt':'Count','mnth':'Month','season':'Season'},inplace=True)



bike.head()
# Plotting the DistPlot for windspeed



fig = plt.figure(figsize=(10,5))

fig.add_subplot(121)

sns.distplot(bike['windspeed'])

plt.title('Windspeed',fontsize=20)



fig.add_subplot(122)

pt = PowerTransformer()

transformed = pt.fit_transform(bike[['windspeed']])

sns.distplot(transformed)

plt.title('Transformed-Windspeed')

plt.show()
# Humidity Variable



fig = plt.figure(figsize=(10,5))

fig.add_subplot(121)

sns.distplot(bike['Humidity'])

plt.title('Humidity',fontsize=20)



fig.add_subplot(122)

pt = PowerTransformer()

transformed = pt.fit_transform(bike[['Humidity']])

sns.distplot(transformed)

plt.title('Transformed-Humidity')

plt.show()
# Temperature Variable



fig = plt.figure(figsize=(10,5))

fig.add_subplot(121)

sns.distplot(bike['Temperature'])

plt.title('Temperature',fontsize=20)



fig.add_subplot(122)

pt = PowerTransformer()

transformed = pt.fit_transform(bike[['Temperature']])

sns.distplot(transformed)

plt.title('Transformed-Temperature')

plt.show()
for i in ['Season','Month','weekday','Weather']:

    print('---------------------------')

    print(bike[i].name)

    print()

    print(bike[i].value_counts())

    print()
def Season_to_text(x):

    if x==1:

        return 'Spring'

    if x==2:

        return 'Summer'

    if x==3:

        return 'Fall'

    else:

        return 'Winter'



    

def weather_sit(x):

    if x==1:

        return 'Clear'

    elif x==2:

        return 'Mist'

    elif x==3:

        return 'Light_Rain'

    else:

        return 'Heavy_Rain'



days = {0:'Sunday',1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday'}

months = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sept',10:'Oct',11:'Nov',12:'Dec'}



    

bike.Season = bike.Season.apply(Season_to_text)

bike.Weather = bike.Weather.apply(weather_sit)

bike.weekday = bike.weekday.apply(lambda x : days.get(x))

bike.Month = bike.Month.apply(lambda x:months.get(x))
bike.head()
bike.Season = bike.Season.apply(lambda x: {'Spring':'Winter','Summer':'Spring','Fall':'Summer','Winter':'Fall'}.get(x))



bike.head()
# Plotting the boxplot for the categorical variables



plt.figure(figsize=(15,20))

for i,j in enumerate(['Season','Weather','Month','Year']):

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    plt.subplot(3,2,i+1)

    sns.boxplot(data=bike,x=j,y='Count')

    plt.title(j +' vs '+ bike.Count.name,fontsize=25)

    plt.xticks(fontsize=15)

    plt.yticks(fontsize=15)

plt.show()
# Plotting the boxplot for the categorical variables



plt.figure(figsize=(20,10),linewidth=0.15,)

for i,j in enumerate(['Month','weekday']):

    plt.subplot(2,2,i+1)

    sns.boxenplot(data=bike,x=j,y='Count',hue='Year')

    plt.title(j +' vs '+ bike.Count.name,fontsize=25)

plt.show()
# Plotting the boxplot for the categorical variables



plt.figure(figsize=(10,10),linewidth=0.55)

for i,j in enumerate(['Season','Month']):

    plt.subplots_adjust(wspace=0.2,hspace=0.3)

    plt.subplot(2,1,i+1)

    sns.barplot(data=bike,x=j,y='Count',hue='Weather')

    plt.title(j +' vs '+ bike.Count.name,fontsize=25)

    plt.legend(loc='best')

    plt.xticks(fontsize=10)

    plt.yticks(fontsize=10)

plt.show()
# Creating dummy variables on the data set



bike1 = pd.get_dummies(bike,drop_first=True)
bike2=bike1.copy()
# Checking the pairplot 



plt.figure(figsize=(20,30))

sns.pairplot(data=bike2,x_vars=['Temperature','casual','registered','Humidity','windspeed','atemp'],y_vars=['Count','atemp'])

plt.show()
# Getting the correlation matrix



bike2[['Temperature','casual','registered','Humidity','windspeed','Count','atemp']].corr().round(2)
# Plotting the heatmap



plt.figure(figsize=(20,10))

sns.heatmap(bike2[['Temperature','casual','registered','Humidity','windspeed','Count','atemp']].corr().round(2),annot=True,center=0.3,linewidths=0.5)

plt.show()
# Removing the redundant and unnecessary variables 



bike2.drop(columns=['instant','dteday','atemp','casual','registered'],inplace=True)
bike2.info()
# Plotting the heatmap



plt.figure(figsize=(20,20))

sns.heatmap(bike2.corr().round(2),annot=True,cmap='RdYlGn',center=0.2)

plt.show()
# Splitting the dataset into train and test 



np.random.seed(0)

train,test = train_test_split(bike2,train_size=0.7,test_size=0.3,random_state=100)
train.head()
# Rescaling the train and test



var = ['Humidity','windspeed','Temperature','Count']



scaler = MinMaxScaler()



train[var] = scaler.fit_transform(train[var])



test[var] = scaler.transform(test[var])
# Dividing train and test into their corresponsing X and Y variables



# Train

y_train = train.pop('Count')

x_train = train



# Test

y_test = test.pop('Count')

x_test = test
print('Shape of x_train :',x_train.shape)

print('Shape of y_train :',y_train.shape)
print('Shape of x_test :',x_test.shape)

print('Shape of y_test :',y_test.shape)
# Model 1 - with all the predictors



x_train_sm = sm.add_constant(x_train)



lr = sm.OLS(y_train,x_train_sm).fit()



lr.summary()
# Initiating the model using sklearn.linear_model



lm = LinearRegression()



lm.fit(x_train,y_train)
# rfe variable taking 15 top features



rfe = RFE(lm,15)

rfe=rfe.fit(x_train,y_train)
# Making the dataframe to choose the important features



df = pd.DataFrame(columns=['Features','Support','Rank'])

df['Features'] = x_train.columns

df['Support'] = rfe.support_

df['Rank'] = rfe.ranking_



# Sorting the dataframe 

df
# Taking the valid columns for model building after doing rfe



col = x_train.columns[rfe.support_]



print(col)
# Building the Model 2



x_train_sm = sm.add_constant(x_train[col])



lr = sm.OLS(y_train,x_train_sm).fit()



y_train_pred = lr.predict(x_train_sm)



lr.summary()
# Function for the VIF values



def vif(x=x_train):

    VIF = pd.DataFrame(columns=['Features','VIF'])

    VIF['Features'] = x.columns

    VIF['VIF']  = [ variance_inflation_factor(x.values,i) for i in range(x.shape[1])]

    return VIF.sort_values(by='VIF',ascending=False)

# VIF Check



vif(x_train[col])
x = x_train[col]



x = x_train[col].drop(columns=['Humidity'])



vif(x)
x = x_train[col]



x = x_train[col].drop(columns=['Humidity','Season_Summer'])



vif(x)
# Building the model 3



x_train_sm = sm.add_constant(x)



lr = sm.OLS(y_train,x_train_sm).fit()



y_train_pred = lr.predict(x_train_sm)



lr.summary()
# lm 4 



x = x_train[col]



x = x_train[col].drop(columns=['Humidity','Season_Summer','Season_Spring'])



vif(x)
# Building the model 4



x_train_sm = sm.add_constant(x)



lr = sm.OLS(y_train,x_train_sm).fit()



y_train_pred = lr.predict(x_train_sm)



lr.summary()
# lm5 



x = x_train[col]



x = x_train[col].drop(columns=['Humidity','Season_Spring','Season_Summer','Month_Nov'])



vif(x)
# Building the model 5



x_train_sm = sm.add_constant(x)



lr = sm.OLS(y_train,x_train_sm).fit()



y_train_pred = lr.predict(x_train_sm)



lr.summary()
# lm 6



x = x_train[col]



x = x_train[col].drop(columns=['Humidity','Season_Summer','Season_Spring','Month_Nov','Month_Dec'])



vif(x)
# Building the model 6



x_train_sm = sm.add_constant(x)



lr = sm.OLS(y_train,x_train_sm).fit()



y_train_pred = lr.predict(x_train_sm)



lr.summary()
# lm 7

x = x_train[col]



x = x_train[col].drop(columns=['Humidity','windspeed','Season_Summer','Season_Spring','Month_Nov','Month_Dec','Month_Jan','Month_Sept'])





# VIF Check



vif(x)
# Building the model 7



x_train_sm = sm.add_constant(x)



lr = sm.OLS(y_train,x_train_sm).fit()



y_train_pred = lr.predict(x_train_sm)



lr.summary()
# Calculating predicted values of y



y_train_pred = lr.predict(x_train_sm)
res= y_train - y_train_pred





# Creating the Distplot for errors

plt.figure(figsize=(10,6))

sns.distplot(res,bins=20)

plt.title('Errors',fontsize=25)

plt.show()
# Making Predictions



X = x_test[x.columns]

X_test_sm = sm.add_constant(X)

y_test_pred = lr.predict(X_test_sm)
x_test.shape

y_test.shape
# Model Evaluation



plt.figure(figsize=(10,8))

sns.scatterplot(x=y_test,y=y_test_pred,color='r')

plt.title('y_test vs y_test_pred',fontsize=25)

plt.xlabel('y_test',fontsize=20)

plt.ylabel('y_test_pred',fontsize=20)

plt.show()
# R2 score 



print('Train R-square : ',round(r2_score(y_train,y_train_pred),3))

print('Test R-square :',round(r2_score(y_test,y_test_pred),3))
# Calculating R2 and adjusted R2 values for training dataset



residuals_train = lr.resid



rss_train = sum(res**2)



tss_train = sum((y_train - np.mean(y_train))**2)



r2_train = 1-rss_train/tss_train



n = x_train.shape[0]



p = len(x.columns)



adj_r2_train = 1 - ((1-r2_train)*(n-1))/(n-p-1)



print('R2 value for the training data',round(r2_train,3))

print('Adjusted R2 value for the training data',round(adj_r2_train,3))
# Calculating R2 and adjusted R2 values for test dataset



residuals_test = y_test - y_test_pred



rss_value = sum(residuals_test**2)



tss_value = sum((y_test - np.mean(y_test))**2)



r2_test_value = 1-rss_value/tss_value



print('R2 value for test data :',round(r2_test_value,3))



n = X.shape[0]



p = len(X.columns)



adj_r2_test = 1 - ((1-r2_test_value)*(n-1))/(n-p-1)



print('Adjusted R2 value for test data :',round(adj_r2_test,3))
# RMSE value



rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))



print('RMSE Value :',rmse)
# Normal distribution of error terms



res = lr.resid



plt.figure(figsize=(20,10))



fig = sm.qqplot(res,fit=True,line='r')



plt.title('Theoretical Qunatiles vs Normal Distribution of residuals')



plt.show()
# Residual plot to check Homoscedasticity



from yellowbrick.regressor import ResidualsPlot

plt.figure(figsize=(10,5))



lm = LinearRegression()



vis = ResidualsPlot(lm).fit(x,y_train)



vis.score(X,y_test)



vis.poof()



plt.show()
# Autocorrelation check



from statsmodels.graphics import tsaplots



fig = tsaplots.plot_acf(lr.resid,alpha=0.05)



plt.show()
# Checking the coefficients



lr.params