# Supress Warnings



import warnings

warnings.filterwarnings('ignore')



# Import the numpy and pandas package



import numpy as np

import pandas as pd



# Data Visualisation



import matplotlib.pyplot as plt 

import seaborn as sns
#Importing dataset

media = pd.DataFrame(pd.read_csv("../input/mediacompany.csv"))

media.head()
# Checking Duplicates

sum(media.duplicated(subset = 'Date')) == 0

# No duplicate values
# Dropping the unwanted column

media = media.drop('Unnamed: 7',axis = 1)
#Let's explore the top 5 rows

media.head()
media.shape
media.info()
media.describe()
# Checking Null values

media.isnull().sum()*100/media.shape[0]

# There are no NULL values in the dataset, hence it is clean.
# Outlier Analysis

fig, axs = plt.subplots(2,2, figsize = (10,5))

plt1 = sns.boxplot(media['Views_show'], ax = axs[0,0])

plt2 = sns.boxplot(media['Visitors'], ax = axs[0,1])

plt3 = sns.boxplot(media['Views_platform'], ax = axs[1,0])

plt4 = sns.boxplot(media['Ad_impression'], ax = axs[1,1])



plt.tight_layout()
# Data preparation
# Converting date to Pandas datetime format

media['Date'] = pd.to_datetime(media['Date'], dayfirst = False )

# Date is in the format YYYY-MM-DD
media.head()
# Let's derive day of week column from date 
media['Day_of_week'] = media['Date'].dt.dayofweek
media.head()
# Target Variable

# Views Show
sns.boxplot(media['Views_show'])
# days vs Views_show

media.plot.line(x='Date', y='Views_show')
# Inference

# we can observe a pattern in the plot.
sns.barplot(data = media,x='Day_of_week', y='Views_show')
# Inference

# we can see that Views are more on 'Sunday' and 'Saturday'(weekends) and decline on subsequent days.
# Hence we can think of another matrix "Weekend" that is 1 for weekends and 0 for weekdays.
di = {5:1, 6:1, 0:0, 1:0, 2:0, 3:0, 4:0}

media['weekend'] = media['Day_of_week'].map(di)
media.head()
sns.barplot(data = media,x='weekend', y='Views_show')
# viewership is higher on weekends.
# plot for Date vs Views_show and days vs Ad_impressions

ax = media.plot(x="Date", y="Views_show", legend=False)

ax2 = ax.twinx()

media.plot(x="Date", y="Ad_impression", ax=ax2, legend=False, color="r")

ax.figure.legend()

sns.scatterplot(data = media, x = 'Ad_impression', y = 'Views_show')
# we can see that the views as well as ad impressions show a weekly pattern.
sns.scatterplot(data = media, x = 'Visitors', y = 'Views_show')
# Inference: Show views are some what proportionately related to Visitors
sns.scatterplot(data = media, x = 'Views_platform', y = 'Views_show')
# Inference: Show views are some what proportionately related to Platform views
sns.barplot(data = media,x='Cricket_match_india', y='Views_show')
# Inference: Show views slightly declines when there is a cricket match.
sns.barplot(data = media,x='Character_A', y='Views_show')
# Inference: Presence of Character A improves the show viewership.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

num_vars = ['Views_show','Visitors','Views_platform','Ad_impression']



media[num_vars] = scaler.fit_transform(media[num_vars])
media.head()
# Let's check the correlation coefficients to see which variables are highly correlated
sns.heatmap(media.corr(),annot = True)
# Putting feature variable to X

X = media[['Visitors','weekend']]



# Putting response variable to y

y = media['Views_show']
from sklearn.linear_model import LinearRegression
# Representing LinearRegression as lm(Creating LinearRegression Object)

lm = LinearRegression()
# fit the model to the training data

lm.fit(X,y)
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_1 = sm.OLS(y,X).fit()

print(lm_1.summary())
# Inference:

# Visitors as well as weekend column are significant.
# Putting feature variable to X

X = media[['Visitors','weekend','Character_A']]



# Putting response variable to y

y = media['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_2 = sm.OLS(y,X).fit()

print(lm_2.summary())
# we have seen that views of today effects views of tomorrow. So to take that in account we will create a Lag variable.
# Create lag variable

media['Lag_Views'] = np.roll(media['Views_show'], 1)

media.head()
media.Lag_Views[0] = 0
media.head()
# Putting feature variable to X

X = media[['Visitors','Character_A','Lag_Views','weekend']]



# Putting response variable to y

y = media['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_3 = sm.OLS(y,X).fit()

print(lm_3.summary())
# Inference:

# It leaves visitor insignificant.
# Putting feature variable to X

X = media[['weekend','Character_A','Views_platform']]



# Putting response variable to y

y = media['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_4 = sm.OLS(y,X).fit()

print(lm_4.summary())
# Putting feature variable to X

X = media[['weekend','Character_A','Visitors']]



# Putting response variable to y

y = media['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_5 = sm.OLS(y,X).fit()

print(lm_5.summary())
# Putting feature variable to X

X = media[['weekend','Character_A','Visitors','Ad_impression']]



# Putting response variable to y

y = media['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_6 = sm.OLS(y,X).fit()

print(lm_6.summary())
# Inference

# we can observe a pattern in the plot.
# Putting feature variable to X

X = media[['weekend','Character_A','Ad_impression']]



# Putting response variable to y

y = media['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_7 = sm.OLS(y,X).fit()

print(lm_7.summary())
#Ad impression in million

media['ad_impression_million'] = media['Ad_impression']/1000000
# Putting feature variable to X

X = media[['weekend','Character_A','ad_impression_million','Cricket_match_india']]



# Putting response variable to y

y = media['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_8 = sm.OLS(y,X).fit()

print(lm_8.summary())
# Putting feature variable to X

X = media[['weekend','Character_A','ad_impression_million']]



# Putting response variable to y

y = media['Views_show']
import statsmodels.api as sm

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X = sm.add_constant(X)

# create a fitted model in one line

lm_9 = sm.OLS(y,X).fit()

print(lm_9.summary())
# Making predictions using the model

X = media[['weekend','Character_A','ad_impression_million']]

X = sm.add_constant(X)

Predicted_views = lm_9.predict(X)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(media.Views_show, Predicted_views)

r_squared = r2_score(media.Views_show, Predicted_views)
print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)
#Actual vs Predicted

c = [i for i in range(1,81,1)]

fig = plt.figure()

plt.plot(c,media.Views_show, color="blue", linewidth=2.5, linestyle="-")

plt.plot(c,Predicted_views, color="red",  linewidth=2.5, linestyle="-")

fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('Views', fontsize=16)                               # Y-label
# Error terms

c = [i for i in range(1,81,1)]

fig = plt.figure()

plt.plot(c,media.Views_show-Predicted_views, color="blue", linewidth=2.5, linestyle="-")

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                      # X-label

plt.ylabel('Views_show-Predicted_views', fontsize=16)                # Y-label
# Making predictions using the model

X = media[['weekend','Character_A','Visitors']]

X = sm.add_constant(X)

Predicted_views = lm_5.predict(X)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(media.Views_show, Predicted_views)

r_squared = r2_score(media.Views_show, Predicted_views)
print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)
#Actual vs Predicted

c = [i for i in range(1,81,1)]

fig = plt.figure()

plt.plot(c,media.Views_show, color="blue", linewidth=2.5, linestyle="-")

plt.plot(c,Predicted_views, color="red",  linewidth=2.5, linestyle="-")

fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('Views', fontsize=16)                               # Y-label
# Error terms

c = [i for i in range(1,81,1)]

fig = plt.figure()

plt.plot(c,media.Views_show-Predicted_views, color="blue", linewidth=2.5, linestyle="-")

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                      # X-label

plt.ylabel('Views_show-Predicted_views', fontsize=16)                # Y-label