#------------Importing EDA Libraries--------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#-------------Importing Machine Learnig Libraries--------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score

#---------------Importing RFE and LinearRegression-------------------------
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


#-------------Handling Warnings--------------------------------------------
import warnings
warnings.filterwarnings('ignore')
# load the dataset
df = pd.read_csv("../input/boom-bikes/day.csv")
# Take a look at the data
df.head()
# Shape----
df.shape
# Data Discription----
df.describe()
# Checking the dtypes and missing values----
df.info()
sns.pairplot(df[['temp','atemp','hum','windspeed','cnt']])     
plt.show
# Plotting boxplots to dipict the affect of each categorical variable on the target variable 'cnt'.
plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'season', y = 'cnt', data = df)
plt.subplot(2,3,2)
sns.boxplot(x = 'yr', y = 'cnt', data = df)
plt.subplot(2,3,3)
sns.boxplot(x = 'holiday', y = 'cnt', data = df)
plt.subplot(2,3,4)
sns.boxplot(x = 'weekday', y = 'cnt', data = df)
plt.subplot(2,3,5)
sns.boxplot(x = 'workingday', y = 'cnt', data = df)
plt.subplot(2,3,6)
sns.boxplot(x = 'weathersit', y = 'cnt', data = df)
plt.show()
# We can see that how the median values varies for each of these categorical variables.
# Boxplot to show how 'cnt' varies on holidays for all the season 
plt.figure(figsize = (10, 5))
sns.boxplot(x = 'season', y = 'cnt', hue = 'holiday', data = df)
plt.show()
# Droping the unnecessary columns
drop_cols = ['instant','dteday','casual','registered']
df.drop(drop_cols,axis = 1,inplace = True)
# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")
plt.show()
# Season--- 
def season(value):
    if value == 1:
        return 'Spring'
    elif value == 2:
        return 'Summer'
    elif value == 3:
        return 'Fall'
    else:
        return 'Winter'

df['Season'] = df['season'].apply(season)


# Year---

def year(value):
    if value == 0:
        return '2018'
    else:
        return '2019'
df['Yr'] = df['yr'].apply(year)

# Weekday---

def weekday(value):
    if value == 0:
        return 'sunday'
    elif value == 1:
        return 'monday'
    elif value == 2:
        return 'tuesday'
    elif value ==3:
        return 'wednesday'
    elif value ==4:
        return 'thursday'
    elif value == 5:
        return 'friday'
    else:
        return 'saturday'
    
df['Weekday'] = df['weekday'].apply(weekday) 

# Weathersit---

def weathersit(value):
    if value == 1:
        return 'clear'
    elif value == 2:
        return 'mist'
    elif value == 3:
        return 'snow'
    else:
        return 'heavy_rain'
    
df['Weathersit'] = df['weathersit'].apply(weathersit)


# Month---

def month(value):
    if value == 1:
        return 'jan'
    elif value == 2:
        return 'feb'
    elif value == 3:
        return 'mar'
    elif value == 4:
        return 'apr'
    elif value == 5:
        return 'may'
    elif value == 6:
        return 'jun'
    elif value == 7:
        return 'jul'
    elif value == 8:
        return 'aug'
    elif value == 9:
        return 'sep'
    elif value == 10:
        return 'oct'
    elif value == 11:
        return 'nov'
    else:
        return 'dec'

df['Month'] = df['mnth'].apply(month) 

# Holiday

def holiday(value):
    if value == 0:
        return 'No_holiday'
    else:
        return 'Yes_holiday'
    
df['Holiday'] = df['holiday'].apply(holiday)

def workingday(value):
    if value == 0:
        return 'No_working'
    else:
        return 'Yes_working'
    
df['Workingday'] = df['workingday'].apply(workingday)

# Drop the columns which has been modified

drop_columns = ['season','yr','weekday','weathersit','mnth','holiday','workingday']
df.drop(drop_columns,axis=1,inplace = True)
# Lets look at the dataset after conversion into categorical variables
df.head()
# Checking the dtypes of the variables
df.info()
# We can see that all the binary and numerical categorical columns have been converted into categorical columns
# Season variable
season = pd.get_dummies(df['Season'],drop_first=True)
season.head()
# Year variable
yr = pd.get_dummies(df['Yr'],drop_first=True)
yr.head()
# Weekday variable
weekday = pd.get_dummies(df['Weekday'],drop_first=True)
weekday.head()
# Weathersit variable
weathersit = pd.get_dummies(df['Weathersit'],drop_first=True)
weathersit.head()
# month variable
month = pd.get_dummies(df['Month'],drop_first=True)
month.head()
# Holiday variable
holiday = pd.get_dummies(df['Holiday'],drop_first=True)
holiday.head()
# Working day variable
workingday = pd.get_dummies(df['Workingday'],drop_first=True)
workingday.head()
# Firsly we will merge them altogether
frames = [df,season,yr,weekday,weathersit,month,holiday,workingday]
df = pd.concat(frames,axis = 1)
# Dataset
df.head()
# Shape
df.shape
# lets drop the columns we used dummy encoding for--
drop_col1 = ['Season','Yr','Weekday','Weathersit','Month','Holiday','Workingday']
df.drop(drop_col1,axis = 1, inplace = True)
# Head
df.head()
# Dataset columns
df.columns
# Shape
df.shape
#------------- Correlation between atemp and other variables
print(round(df.atemp.corr(df.cnt),4))
print(round(df.atemp.corr(df.hum),4))
print(round(df.atemp.corr(df.windspeed),4))

# We can see that actual temp has a positive correlation with the cnt variable 
#------------- Correlation between hum and other variables
print(round(df.hum.corr(df.cnt),4))
print(round(df.hum.corr(df.Summer),4))
print(round(df.hum.corr(df.windspeed),4))

# We can see that as the humidity increases the cnt decreases which states that bikers do not like the 
# humid weather
#------------- Correlation between windspeed and other variables
print(round(df.windspeed.corr(df.cnt),4))
print(round(df.windspeed.corr(df.atemp),4))
print(round(df.windspeed.corr(df.Winter),4))

# We can see that as the windpeed increases the cnt decreases which states that bikers do not like the 
# windy climate
# Spliting the data into train and test datasets
np.random.seed(0)
df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)
#Shape of train and test data
print(df_train.shape)
print(df_test.shape)
# Scaling the some of the features
var = ['temp','atemp','hum','windspeed']
scaler = MinMaxScaler()
df_train[var] = scaler.fit_transform(df_train[var])
# Dividing the training dataset into X and y.
y_train = df_train.pop('cnt')
X_train = df_train
# Building a model using statsmodel
# Add a constant
X_train_lm = sm.add_constant(X_train)
# Create a first fitted model
lr_1 = sm.OLS(y_train, X_train_lm).fit()
# First model
lr_1.summary()
# Running RFE with the output number of the variable equal to 15
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 15)             # running RFE
rfe = rfe.fit(X_train, y_train)
# List of 15 columns selected by RFE
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# Columns selected by RFE
col = X_train.columns[rfe.support_]
col
# Columns rejected by RFE
X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]
# Adding a constant variable  
X_train_rfe = sm.add_constant(X_train_rfe)
# Second model
lm_2 = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model
# Summary statistics
lm_2.summary()
# Calculate the VIFs for the new model
vif = pd.DataFrame()
X = X_train[col]
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train[col].drop(["hum"], axis = 1)
X_train_lm = sm.add_constant(X_train_new)
# Third model
lm_3 = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
lm_3.summary()
# Calculate the VIFs for the new model
vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Surely the VIF values have come down
X_train_new = X_train[col].drop(['temp','Yes_working'], axis = 1)
X_train_lm = sm.add_constant(X_train_new)
# Fourth model
lm_4 = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
lm_4.summary()
X_train_new = X_train[col].drop(['Yes_working','hum'], axis = 1)
X_train_lm = sm.add_constant(X_train_new)
# Fifth model
lm_5 = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
lm_5.summary()
# Calculate the VIFs for the new model
vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# We will try to find the variables humidity is correlated with and try removing them.
#------------- Correlation between hum and other variables
print(round(df.hum.corr(df.cnt),4))
print(round(df.hum.corr(df.Summer),4))
print(round(df.hum.corr(df.windspeed),4))
print(round(df.hum.corr(df.Spring),4))
print(round(df.hum.corr(df.jan),4))
print(round(df.hum.corr(df.jul),4))
print(round(df.hum.corr(df.sep),4))
print(round(df.hum.corr(df.saturday),4))
print(round(df.hum.corr(df.Yes_holiday),4))
X_train_new = X_train[col].drop(['hum','jul','jan','Yes_holiday','saturday'], axis = 1)
X_train_lm = sm.add_constant(X_train_new)
lm_6 = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
# Final model
lm_6.summary()
# Calculate the VIFs for the new model
vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Pairplot of final variables
sns.pairplot(df[['temp','atemp','hum','windspeed','Yes_working','Spring','2019','Summer','Winter','mist','sep','snow','cnt']]) 
y_train_price = lm_6.predict(X_train_lm)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)    
var = ['temp','atemp','hum','windspeed']

df_test[var] = scaler.transform(df_test[var])
y_test = df_test.pop('cnt')
X_test = df_test
# Now let's use our model to make predictions.

# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)
# Making predictions
y_pred = lm_6.predict(X_test_new)
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label
# Calculating R2 value for test data
r2 = r2_score(y_test,y_pred)
print(r2)