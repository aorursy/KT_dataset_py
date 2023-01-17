import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings('ignore')
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor
import sklearn

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
#  Reading Dataset

df = pd.read_csv('../input/boombikes/day.csv')

df.head()
df.shape
df.info()
df.drop('instant', axis=1, inplace=True)
df.dteday = pd.to_datetime(df.dteday)
# Year

df.yr = df.yr.map({1: 2019, 0 : 2018})

# Season

df.season = df.season.map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})

# Month

df.mnth = df.mnth.map({1:'Jan',2:'Feb',3:'March',4:'April',

                            5:'May',6:'June',7:'July',8:'Aug',9:'Sep',

                            10:'Oct',11:'Nov',12:'Dec'})

# weekday

df.weekday =df.weekday.map({1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thrusday',

                                  5:'Friday',6:'Saturday',0:'Sunday',})
df.weathersit =df.weathersit.map({1:'Clear', 2:'Cloudy', 3:'LightSnow Rain', 4:'HeavysSnow Rain'})
print("Season :- {}".format(df.season.unique()))

print("Years :- {}".format(df.yr.unique()))

print("Months :- {}".format(df.mnth.unique()))

print("Weekday :- {}".format(df.weekday.unique()))

print("Weather Situation :- {}".format(df.weathersit.unique()))
df.head()
df.rename(columns = {'dteday':'date','yr':'year','mnth':'month','hum':'humidity','cnt':'count'}, inplace = True)
df.info()
# Statistical Summary

df.describe()
df.groupby('date').agg({'date':'count'}).shape
print('dataset starting date = {}'.format(df.date.min().date()))

print('dataset ending date = {}'.format(df.date.max().date()))

s = df.date.min().date()

e = df.date.max().date()

print("No's of days between max and min date = {}".format((e-s).days))
total_days = df.groupby(['weathersit','workingday']).agg({'workingday':'count'})

total_days.rename(columns={'workingday':'count of days'}, inplace = True)
total_days.pivot_table(index=['weathersit','workingday'],

               margins=True,

               margins_name='total',  # defaults to 'All'

               aggfunc=sum)

# total_days
plt.figure(figsize=(16,6))

sns.swarmplot(x="weathersit", y="count", hue="workingday",

                   data=df, palette="Set1", dodge=True)

plt.title('Climatic Condition Vs Count of the Renters')

plt.show()
#  Though we have seen the spread of the data lets also explore its density 

df.weathersit.value_counts(normalize=True)
df.weekday.value_counts(normalize=True).apply(lambda x:str(round(x*100,2))+'%')
%matplotlib inline



fig, axes = plt.subplots(1, 2,sharey=True,figsize=(15,8))

plt.suptitle('WEEKDAYS VS SPREAD OF RENTERS')





sns.swarmplot(x="weekday", y="count", hue='weathersit',

                   data=df[(df['weathersit']!= 'LightSnow Rain') & (~df['weekday'].isin(['Saturday', 'Sunday']))], 

              palette=['Green','Blue'], dodge=True,ax=axes[0])





sns.swarmplot(x="weekday", y="count", hue='weathersit',

                   data=df[(df['weathersit']!= 'LightSnow Rain') & (df.weekday.isin (['Saturday', 'Sunday']))],

              palette=['Blue','Green'], dodge=True ,ax=axes[1])

plt.xlabel('Weekend')



plt.show()
rented_user = df.groupby('month', as_index =False,sort=False).agg({'count':'sum','registered':'sum','casual':'sum'})

rented_user
plt.figure(figsize=(10,5))

sns.lineplot(rented_user.month,rented_user.casual,color='b',label='Casual User',sort=False)

sns.lineplot(rented_user.month,rented_user.registered,color='g',dashes='-',label='Registered User',sort=False)

plt.ylabel('Types of Users and their respective intractions')

plt.xlabel('Months')

plt.title('Casual User vs Registered User Monthly')

plt.show()
plt.figure(figsize=(16,6))

# sns.lineplot(x="mnth", y="cnt", data=df, color='red')

sns.violinplot(x="month", y="count", data=df)

sns.boxplot(x="month", y="count", data=df)

plt.title("Renters Density Across Months")

plt.show()
#Pairplot for numeric variables

sns.pairplot(df, vars=["temp",'atemp', 'windspeed',"humidity",'casual','registered','count'])

plt.show()
sns.pairplot(df, x_vars=["temp",'atemp', 'windspeed',"humidity",'casual','registered'],y_vars=['count'],hue='year')

plt.show()
corr = df.corr()

# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap='YlGnBu', annot = True)



plt.show()
#drop unnecessary columns

df=df.drop(['date','casual', 'registered','atemp'], axis=1)

df.head()
print('The columns which we have are')

print(list(df.columns))
print('Checking the levels of each columns')

print(df.select_dtypes(include='object').nunique().sort_values())
# Season

print('storing season dummy variables into cat1')

cat1 = pd.get_dummies(df.season, drop_first = True)

cat1.head()
# month

print('storing month dummy variables into cat2')

cat2 = pd.get_dummies(df.month,drop_first=True)

cat2.head()
# weekday

print('    storing weekday dummy variables into cat3')

cat3 = pd.get_dummies(df.weekday,drop_first=True)

cat3.head()
# weathersit

print('storing weathersit dummy variables into cat4')

cat4 = pd.get_dummies(df.weathersit,drop_first=True)

cat4.head()
print('original dataset')
#year

df.year = df.year.map({2018:0,2019:1})

df.head()
df_model = pd.concat((df,cat1,cat2,cat3,cat4), axis=1)
df_model.drop(['season', 'month', 'weekday', 'weathersit'], axis =1, inplace = True)
print('Conctenating the cat1 ,cat2,cat3,cat4 & droping  table as well as droping unwanted repeated columns')

df_model.nunique().sort_values()
print('The Final Model Dataset')
df_model.head()
df_model_train, df_model_test = train_test_split(df_model, train_size = 0.7, test_size = 0.3, random_state = 100)
print('Shape of Train Dataset {}'.format(df_model_train.shape))

df_model_train.head()
print('Shape of Test Dataset {}'.format(df_model_test.shape))

df_model_test.head()
scaler = MinMaxScaler()

variables = ['temp','humidity', 'windspeed']

df_model_train[variables] =scaler.fit_transform(df_model_train[variables])
df_model_train.head()
print('Statistical Summary')

df_model_train.describe()
# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (25,20))

ax =sns.heatmap(df_model_train.corr(), annot = True, cmap="Greens", fmt='.2g',linewidths=1)

plt.yticks(size=15)

plt.xticks(size=15)

# plt.show()

for lab in ax.get_yticklabels():

    text =  lab.get_text()

    if text == 'count': # lets highlight row 2

        # set the properties of the ticklabel

        lab.set_weight('bold')

        lab.set_size(20)

        lab.set_color('purple')



for lab in ax.get_xticklabels():

    text =  lab.get_text()

    if text == 'count': # lets highlight row 2

        # set the properties of the ticklabel

        lab.set_weight('bold')

        lab.set_size(20)

        lab.set_color('purple')
y_train = df_model_train.pop('count')
X_train = df_model_train
print('Count as Target variable (ytrain)')

pd.DataFrame(y_train.head())
print('Xtrain as')

X_train.head()
# Add Constant

X_train_sm = sm.add_constant(X_train)

# create a fitted model in one line

lr_1 = sm.OLS(y_train,X_train_sm).fit()

print('Model 1 summary')

print(lr_1.summary())
from sklearn.feature_selection import RFE

lm = LinearRegression()

rfe = RFE(lm, 15)

rfe = rfe.fit(X_train, y_train)
#List of variables selected

list(zip(X_train.columns,rfe.support_,rfe.ranking_))
#Columns where RFE support is True

feature = X_train.columns[rfe.support_]

feature = list(feature)

feature
#Columns where RFE support is False

col = X_train.columns[~rfe.support_]

col =list(col)
col
# Adding constant as statsmodel wont provide constant by default and assumes line to be passed through the origin

X_train_2 = sm.add_constant(X_train[feature])
lr = sm.OLS(y_train,X_train_2).fit()

print(lr.summary())
# Creating dataframe to store a VIF information

X_train_vif = X_train[feature]

vif1=pd.DataFrame()

vif1['features'] = X_train_vif.columns

vif1['VIF'] = [variance_inflation_factor(X_train[feature].values,i) for i in range(15)]

vif1['VIF'] = round(vif1['VIF'], 2)

vif1 = vif1.sort_values(by = "VIF", ascending = False).reset_index(drop=True)

vif1
feature.remove('Jan')
def get_model(cnt,feature,X_train,final=False):

    global X_train_sm1

    global lr

    global vif

    # add_constant

    X_train_sm1 = sm.add_constant(X_train[feature])

    # fit dtraight line

    lr = sm.OLS(y_train, X_train_sm1).fit()

    # print_summary

    print(lr.summary())

    vif='vif'+str(cnt)

    vif=pd.DataFrame()

    vif['features'] = X_train[feature].columns

    vif['VIF'] = [variance_inflation_factor(X_train[feature].values,i) for i in range(X_train[feature].shape[1])]

    vif['VIF'] = round(vif['VIF'],2)

    vif= vif.sort_values('VIF', ascending=False).reset_index(drop=True)

    print('-------------------------------------------------------------------------------')

    print('VIF'.center(75))

    print('===============================================================================')

    print(vif)
get_model(3,feature,X_train)
feature.remove('humidity')

get_model(4,feature,X_train)
feature.remove('temp')

get_model(5,feature,X_train)
feature.append('temp')

feature.remove('July')

get_model(6,feature,X_train)
feature.remove('holiday')

get_model(7,feature,X_train)
feature.remove('windspeed')

get_model(8,feature,X_train)
feature.append('windspeed')

feature.remove('temp')

get_model(9,feature,X_train)
feature.append('temp')

feature.remove('windspeed')

get_model(8,feature,X_train)
# residual = y_pred - y_train

# y_train = y_train.reshape(y_train.shape[0])
y_train.shape
X_train_sm1.head(3)
y_pred = lr.predict(X_train_sm1)



fig = plt.figure()

sns.distplot((y_train - y_pred), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)

plt.show()
residual = y_train - y_pred

sns.scatterplot(y_train,residual)

plt.plot(y_train,(y_train - y_train), '-r')

plt.xlabel('Count')

plt.ylabel('Residual')

plt.show()
sm.qqplot((y_train - y_pred), fit=True, line='45')

plt.title('Q-Q Plot')

plt.show()
DW_value = round(sm.stats.stattools.durbin_watson((y_train - y_pred)),4)

print('The Durbin-Watson value for Final Model8 is',DW_value)
# Calling VIF from the function for the final model define within the function

vif
df_model_test.head()
df_model_test.columns
df_model_test[variables] = scaler.transform(df_model_test[variables])
df_model_test.head()
y_test = df_model_test.pop('count')

X_test = df_model_test
# Add Constant

X_test_sm = sm.add_constant(X_test[feature])

# create a fitted model in one line

lr_test = sm.OLS(y_test,X_test_sm).fit()

y_test_pred = lr_test.predict(X_test_sm)
#MSE

print('Mean Square Error: {}'.format(np.sqrt(mean_squared_error(y_test, y_test_pred))))



r_squared = r2_score(y_test, y_test_pred)

print('R Square: {}'.format(r_squared))



Adj_r2=1-(1-0.8201349110540019)*(11-1)/(11-1-1)

print('Adj. R Square: {}'.format(Adj_r2))
# Plotting y_test and y_pred to understand the spread.

plt.figure(figsize=(15,6))

plt.scatter(y_test,y_test_pred,color='blue')

plt.title('y_test vs y_pred', fontsize=15) 

plt.xlabel('y_test', fontsize=18) 

plt.ylabel('y_pred', fontsize=16)

plt.show()
from math import sqrt

round(sqrt(mean_squared_error(y_test, y_test_pred)),4)
r_squared = r2_score(y_test, y_test_pred)

print('R Square: {}'.format(r_squared))



Adj_r2=1-(1-0.8201349110540019)*(11-1)/(11-1-1)

print('Adj. R Square: {}'.format(Adj_r2))
pd.DataFrame({"Measurement":['R Square','Adj. R Square'],"Train":[0.826,0.823],"Test":[0.82, 0.80]})
#RMSE

print('Root Mean Square Error: {}'.format(np.sqrt(mean_squared_error(y_test, y_test_pred))))

#AMSE

print('Mean Absolute Error: {}'.format(np.sqrt(mean_absolute_error(y_test, y_test_pred))))
#Regression plot

plt.figure(figsize=(15,6))

sns.regplot(x=y_test, y=y_test_pred, ci=68, fit_reg=True,scatter_kws={"color": "blue"}, line_kws={"color": "red"})

plt.title('y_test vs y_pred', fontsize=20)

plt.xlabel('y_test', fontsize=18) 

plt.ylabel('y_pred', fontsize=16)  

plt.show()