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
#Importing libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
#Reading csv data file



df = pd.read_csv('../input/bikes-data/day.csv')



# Checking first five rows of the dataframe

df.head()
#Inspecting dataframe



print(df.shape)
df.describe()
#checking for datatypes

df.info()
# Removing unnecessary columns from the dataframe which are not going to contribute in building model.



cols_to_be_dropped = ['instant', 'dteday','casual','registered']



new_df = df.drop(cols_to_be_dropped, axis =1)



new_df.head()
#Visualising the Data



#Visualising numeric varaibles



sns.pairplot(new_df)

plt.show()
feature_cols = ['season','yr','mnth','holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum','windspeed']

# multiple scatter plots in Seaborn

sns.pairplot(new_df, x_vars=feature_cols, y_vars='cnt', kind='reg')
#Here we can observe that 'temp' and 'atemp' are having linear relationship of points with respect to 'cnt'



#Visualising categorical variables



plt.figure(figsize=(20, 12))

plt.subplot(3,3,1)

sns.boxplot(x = 'season', y = 'cnt', data = new_df)

plt.subplot(3,3,2)

sns.boxplot(x = 'yr', y = 'cnt', data = new_df)

plt.subplot(3,3,3)

sns.boxplot(x = 'mnth', y = 'cnt', data = new_df)

plt.subplot(3,3,4)

sns.boxplot(x = 'holiday', y = 'cnt', data = new_df)

plt.subplot(3,3,5)

sns.boxplot(x = 'weekday', y = 'cnt', data = new_df)

plt.subplot(3,3,6)

sns.boxplot(x = 'workingday', y = 'cnt', data = new_df)

plt.subplot(3,3,7)

sns.boxplot(x = 'weathersit', y = 'cnt', data = new_df)

plt.show()



print('By the categorical varaibales we can infer that the count distribution in the year 2019 is better than the year 2018 with the hightest count of 8714. ⦁ Further we can infer that bikes are least rented in spring and increse in summer. After summer, there is a sudden increase in bike renting during fall with a slight decline in rents during winter. ⦁ Ridership is based on weather as well. When weather is clear, number of ridership is high. There is a slight decline in rents during mist + cloudy weather folowing with light snow, light rain + thunderstorm + scattered clouds, light rain + scattered clouds weather. And there is not a single rent placed during heavy Rain + Ice Pallets + thunderstorm + mist, snow + fog weather. ⦁ More we can say that the bikes are rented more during working days as compared to holidays. ⦁ It can be seen that bike rents are low during January with an increasing graph till june. Then the rent cont average is around 5000 from June to October and then again there in a decrease in counts till the year end. ⦁ On daily basis we can observe that the average is around 4200 bike rent counts.We can also visualise some of these categorical features parallely by using the hue argument')
plt.figure(figsize = (10, 5))

sns.boxplot(x = 'season', y = 'cnt', hue = 'workingday', data = new_df)

plt.show()
#Data preparation



#Mapping varaibles for season.



new_df[['season']] = new_df[['season']].apply(lambda x: x.map({1:'spring',2:'summer', 3:'fall', 4:'winter'}))



new_df.head()

#Mapping varaibles for mnth



new_df[['mnth']] = new_df[['mnth']].apply(lambda x: x.map({1:'Januaray',2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November',12:'December'}))



new_df.head()
#Mapping varaibles for weekday.



new_df[['weekday']] = new_df[['weekday']].apply(lambda x: x.map({0:'Sunday',1:'Monday',2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday', 6:'Saturday' }))



new_df.head()
#Mapping varaibles for weekday.



new_df[['weathersit']] = new_df[['weathersit']].apply(lambda x: x.map({ 1:'Clear',2:'Mist', 3:'Light Rain ', 4:'Heavy Rain'}))



new_df.head()
#Dummy Variables



# Get the dummy variables for the feature 'season' and store it in a new variable - season



season = pd.get_dummies(new_df['season'], drop_first=True)



# Check how 'season' looks like

season.head()
# Get the dummy variables for the feature 'mnth' and store it in a new variable - mnth



mnth = pd.get_dummies(new_df['mnth'], drop_first=True)



# Check how 'season' looks like

mnth.head()
# Get the dummy variables for the feature 'weekday' and store it in a new variable - weekday



weekday = pd.get_dummies(new_df['weekday'], drop_first=True)



# Check how 'season' looks like

weekday.head()
# Get the dummy variables for the feature 'weathersit' and store it in a new variable - weathersit



weathersit = pd.get_dummies(new_df['weathersit'], drop_first=True)



# Check how 'season' looks like

weathersit.head()
#concat dunny varaibles with the dataframe





new_df = pd.concat([season,mnth,weekday,weathersit,new_df], axis= 1)



new_df.head()
# Drop varaibles for which we have created the dummies for building model





new_df.drop(['season','mnth','weekday','weathersit'], axis = 1, inplace = True)



new_df
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize= (20,15))

sns.heatmap(new_df.corr(), annot = True, cmap="YlGnBu")

plt.show()
new_df.head()
#Dropping one of 'atemp' and 'temp' 



new_df.drop(['temp'], axis = 1, inplace=True)



new_df.head()

#Splitting the Data into Training and Testing Sets



from sklearn.model_selection import train_test_split



df_train, df_test = train_test_split(new_df, train_size = 0.8 , random_state =100)
df_train.shape
df_test.shape
#Rescaling the Features



#We will use MinMaxScaler to rescale the variables



from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'Binary' and 'dummy' variables

num_vars = ['atemp', 'hum', 'windspeed']



df_train[num_vars] = scaler.fit_transform(df_train[num_vars])



df_train.head()
df_train.describe()
#Dividing into X and Y sets for the model building



y_train = df_train.pop('cnt')

X_train = df_train
#Building our model

#This time, we will be using the LinearRegression function from SciKit Learn for its compatibility with

#RFE (which is a utility from sklearn)



# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 10

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 15)             # running RFE

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
#Building model using statsmodel for the detailed statistics

#Model 1



# Creating X_test dataframe with RFE selected variables

X_train_lm = X_train[col]



# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_lm)



lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model



#Let's see the summary of our linear model

print(lm.summary())
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

X_train_vif = X_train[col]

vif['Features'] = X_train_vif.columns

vif['VIF'] = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#Model 2



# Dropping highly correlated variables and insignificant variables



X = X_train_vif.drop('holiday', 1)



# Build a third fitted model

X_train_lm = sm.add_constant(X)



lr_2 = sm.OLS(y_train, X_train_lm).fit()



# Print the summary of the model

print(lr_2.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#Model 3



# Dropping highly correlated variables and insignificant variables



X = X_train_vif.drop(['holiday','hum'], 1)



# Build a third fitted model

X_train_lm = sm.add_constant(X)



lr_2 = sm.OLS(y_train, X_train_lm).fit()



# Print the summary of the model

print(lr_2.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#Model 4



# Dropping highly correlated variables and insignificant variables



X = X_train_vif.drop(['holiday','hum','atemp'], 1)



# Build a third fitted model

X_train_lm = sm.add_constant(X)



lr_2 = sm.OLS(y_train, X_train_lm).fit()



# Print the summary of the model

print(lr_2.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#Model 5



# Dropping highly correlated variables and insignificant variables



X = X_train_vif.drop(['holiday','hum','winter','atemp'], 1)



# Build a third fitted model

X_train_lm = sm.add_constant(X)



lr_2 = sm.OLS(y_train, X_train_lm).fit()



# Print the summary of the model

print(lr_2.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
print("Now as you can see, the VIFs and p-values both are within an acceptable range. So we go ahead and make our predictions using this model only.")
#Residual Analysis of the train data

#So, now to check if the error terms are also normally distributed (which is infact, one of the major 

#assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.



y_train_cnt = lr_2.predict(X_train_lm)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_cnt), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)  
#Making Predictions Using the Final Model

#Now that we have fitted the model and checked the normality of error terms, it's time to go ahead and make 

#predictions using the final



# Apply scaler() to all the columns except the 'Binary' and 'dummy' variables

num_vars = ['atemp', 'hum', 'windspeed']



df_test[num_vars] = scaler.transform(df_test[num_vars])



df_test.describe()
#Dividing into X_test and y_test



y_test = df_test.pop('cnt')

X_test = df_test
# Creating X_test_lm dataframe by dropping variables from X_test_m4



X_test_lm = X_test[col]

X_test_lm.shape
X_test_lm = X_test_lm.drop(['holiday','hum','atemp','winter'], axis = 1)

X_test_lm.head()
# Adding constant variable to test dataframe

X_test_lm = sm.add_constant(X_test_lm)
# Making predictions using the fourth model



y_pred_lm = lr_2.predict(X_test_lm)
y_pred_lm.head()
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
#Looking at the RMSE



#Returns the mean squared error; we'll take a square root

np.sqrt(mean_squared_error(y_test, y_pred_lm))
#Checking the R-squared on the test set



r_squared = r2_score(y_test, y_pred_lm)

r_squared
#Model Evaluation



# Plotting y_test and y_pred=lm to understand the spread



fig = plt.figure()

plt.scatter(y_test, y_pred_lm)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)   
#Here the end of the linear regression with model evaluation.

#The significant varaibles which affect the output are the varaibles that have higher values of coefficient in the 

#final model of the summery i.e. in model 5.