import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import datetime as dt
df = pd.read_csv(r'../input/media-company/mediacompany.csv')
df
df.info()
#changing the date into date data type for further processing

df.Date = pd.to_datetime(df.Date)
df

#dropping the undesired column

df.dropna(axis = 1, how = 'any', inplace =True)
#Exploring different days that can impact the views of show.

#dayofweek gives the day of the week. The day of the week with Monday=0, Sunday=6.



df['day_of_week'] = df.Date.dt.dayofweek
d0 = pd.to_datetime('2017-2-28', format= '%Y-%m-%d')
# Deriving "days since the show started"

from datetime import date



d1 = df.Date

delta = d1 - d0

df['day']= delta
df
# Cleaning days

df['day'] = df['day'].astype(str)

df['day'] = df['day'].map(lambda x: x[0:2])

df['day'] = df['day'].astype(int)
df
#day vs views graph

df.plot(x= 'day', y = 'Views_show')
# days vs Ad_impression graph

df.plot(x= 'day', y = 'Ad_impression')
# plot for days vs Views_show and days vs Ad_impressions



fig = plt.figure()

host = fig.add_subplot(111)



par1 = host.twinx()

par2 = host.twinx()



host.set_xlabel("Day")

host.set_ylabel("View_Show")

par1.set_ylabel("Ad_impression")



color1 = plt.cm.viridis(0)

color2 = plt.cm.viridis(0.5)

color3 = plt.cm.viridis(.9)



p1, = host.plot(df.day,df.Views_show, color=color1,label="View_Show")

p2, = par1.plot(df.day,df.Ad_impression,color=color2, label="Ad_impression")



lns = [p1, p2]

host.legend(handles=lns, loc='best')



# right, left, top, bottom

par2.spines['right'].set_position(('outward', 60))      

# no x-ticks                 

par2.xaxis.set_ticks([])

# Sometimes handy, same for xaxis

#par2.yaxis.set_ticks_position('right')



host.yaxis.label.set_color(p1.get_color())

par1.yaxis.label.set_color(p2.get_color())



plt.savefig("pyplot_multiple_y-axis.png", bbox_inches='tight')
# Putting feature variable to X

X = df[['Visitors','day_of_week']]



# Putting response variable to y

y = df['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_1 = sm.OLS(y,X).fit()

print(lm_1.summary())
#setting a feature column for weekend 

li =[]

for i in (df.day_of_week):

    if (i == 5) or (i==6):

        li.append(1)

    else:

        li.append(0)

    



df['weekend'] = li
df
# Putting feature variable to X

X = df[['Visitors','weekend']]



# Putting response variable to y

y = df['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_2 = sm.OLS(y,X).fit()

print(lm_2.summary())
# Putting feature variable to X

X = df[['Visitors','weekend','Character_A']]



# Putting response variable to y

y = df['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_3 = sm.OLS(y,X).fit()

print(lm_3.summary())
# Creating lag feature so see if previous day's views have an impact on next day's views

df['Lag_Views'] = np.roll(df['Views_show'], 1)

df.Lag_Views.replace(108961,0, inplace=True)
df.head()
# Putting feature variable to X

X = df[['Visitors','Character_A','Lag_Views','weekend']]



# Putting response variable to y

y = df['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_4 = sm.OLS(y,X).fit()

print(lm_4.summary())
import seaborn as sns
plt.figure(figsize = (20,10))        # Size of the figure

sns.heatmap(df.corr(),annot = True)
# Putting feature variable to X

X = df[['weekend','Character_A','Views_platform']]



# Putting response variable to y

y = df['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_5 = sm.OLS(y,X).fit()

print(lm_5.summary())
# Putting feature variable to X

X = df[['weekend','Character_A','Visitors']]



# Putting response variable to y

y = df['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_6 = sm.OLS(y,X).fit()

print(lm_6.summary())
# Putting feature variable to X

X = df[['weekend','Character_A','Visitors','Ad_impression']]



# Putting response variable to y

y = df['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_7 = sm.OLS(y,X).fit()

print(lm_7.summary())
# Putting feature variable to X

X = df[['weekend','Character_A','Ad_impression']]



# Putting response variable to y

y = df['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_8 = sm.OLS(y,X).fit()

print(lm_8.summary())
#Ad impression in million

df['ad_impression_million'] = df['Ad_impression']/1000000
# Putting feature variable to X

X = df[['weekend','Character_A','ad_impression_million','Cricket_match_india']]



# Putting response variable to y

y = df['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_9 = sm.OLS(y,X).fit()

print(lm_9.summary())
# Putting feature variable to X

X = df[['weekend','Character_A','ad_impression_million']]



# Putting response variable to y

y = df['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_10 = sm.OLS(y,X).fit()

print(lm_10.summary())
# Putting feature variable to X

X = df[['weekend','ad_impression_million']]



# Putting response variable to y

y = df['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_11 = sm.OLS(y,X).fit()

print(lm_11.summary())


X = df[['weekend','Character_A','ad_impression_million']]

X = sm.add_constant(X)

Predicted_views = lm_10.predict(X)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(df.Views_show, Predicted_views)

r_squared = r2_score(df.Views_show, Predicted_views)
print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)
#Actual vs Predicted

c = [i for i in range(1,81,1)]

fig = plt.figure()

plt.plot(c,df.Views_show, color="blue", linewidth=2.5, linestyle="-")

plt.plot(c,Predicted_views, color="red",  linewidth=2.5, linestyle="-")

fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('Views', fontsize=16)                               
# Error terms

c = [i for i in range(1,81,1)]

fig = plt.figure()

plt.plot(c,df.Views_show-Predicted_views, color="blue", linewidth=2.5, linestyle="-")

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                      # X-label

plt.ylabel('Views_show-Predicted_views', fontsize=16)                # Y-label
X = df[['weekend','Character_A','ad_impression_million']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.3, random_state=1)
lm.fit(X_train,y_train)
print('Training accuracy=',lm.score(X_train,y_train)*100)
pred = lm.predict(X_test)
from sklearn import metrics

from sklearn.metrics import accuracy_score

print('Prediction accuracy =',metrics.explained_variance_score(y_test, pred)*100)
X = df.drop(['Views_show','Date'], axis =1)
y = df.Views_show
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.3, random_state=1)
lm.fit(X_train,y_train)
print('Training accuracy=',lm.score(X_train,y_train)*100)
pred = lm.predict(X_test)
from sklearn import metrics

from sklearn.metrics import accuracy_score

print('Prediction accuracy =',metrics.explained_variance_score(y_test, pred)*100)
lm.coef_
feature_names = X.columns.values

summary_table = pd.DataFrame(columns = ['Feature_names'], data = feature_names)

summary_table['coeff']= np.transpose(lm.coef_)

summary_table



summary_table.index = summary_table.index +1

summary_table.iloc[0]= ['Intercept', lm.intercept_]









summary_table.sort_index()