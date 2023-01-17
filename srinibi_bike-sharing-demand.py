#Supress warnings 

import warnings 

warnings.filterwarnings('ignore')
#importing libraries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import statsmodels.api as sm

from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression 



#Import the bike sharing dataset

sharing=pd.read_csv(r'../input/bike-sharing/day.csv')


sharing.head()


sharing.info()


sharing.describe()


sharing.shape


round(100*(sharing.isnull().sum()/len(sharing.index)),2).sort_values(ascending=False)
round((sharing.isnull().sum(axis=1)/122)*100,2).sort_values(ascending=False)


sharing_dup=sharing 

sharing_dup.drop_duplicates(subset=None,inplace=True)

sharing_dup.shape


sharing_dummy=sharing.iloc[:,1:16]



for col in sharing_dummy:

    print(sharing_dummy[col].value_counts(),'/n')


sharing.columns

sharing_new=sharing[['season', 'yr', 'mnth', 'holiday', 'weekday',

       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed'

       , 'cnt']]

sharing_new.info()




category_cols=['mnth','weekday','season','weathersit']

for cols in category_cols:

    sharing_new[cols]=sharing_new[cols].astype('category')


sharing_new.info()
#Visualising the Numerical variables in the dataframe 

# Create a new dataframe of only numeric variables:



sharing_new_num=sharing_new[[ 'temp', 'atemp', 'hum', 'windspeed','cnt']]



sns.pairplot(sharing_new_num, diag_kind='kde')

plt.show()
#Visualising the Categorical Variables 

plt.figure(figsize=(25, 10))

plt.subplot(2,3,1)

sns.boxplot(x = 'season', y = 'cnt', data = sharing_new)

plt.subplot(2,3,2)

sns.boxplot(x = 'mnth', y = 'cnt', data = sharing_new)

plt.subplot(2,3,3)

sns.boxplot(x = 'weathersit', y = 'cnt', data = sharing_new)

plt.subplot(2,3,4)

sns.boxplot(x = 'holiday', y = 'cnt', data = sharing_new)

plt.subplot(2,3,5)

sns.boxplot(x = 'weekday', y = 'cnt', data = sharing_new)

plt.subplot(2,3,6)

sns.boxplot(x = 'workingday', y = 'cnt', data = sharing_new)

plt.show()
# Visualising how weathersituation impacts in bikesharing



plt.figure(figsize = (8,4))

sns.lineplot(x = 'mnth', y = 'cnt', data = sharing_new, estimator = np.average, hue = 'weathersit', palette = 'coolwarm')

plt.ylabel('Average Count')

plt.show()

#visualising how season impacts bikesharing 

plt.figure(figsize = (8,4))

season= sharing_new[['season','cnt']].groupby(['season']).sum().reset_index().plot(kind='bar',

                                       legend = False, title ="Counts of Bike Sharing by season", 

                                         stacked=True, fontsize=12)

season.set_xlabel("season", fontsize=12)

season.set_ylabel("Count", fontsize=12)

season.set_xticklabels(['spring','summer','fall','winter'])
#Visualisation of Bikesharing in weekday and holiday 

plt.figure(figsize = (10, 5))

sns.boxplot(x = 'weekday', y = 'cnt', hue = 'holiday', data = sharing_new)

plt.show()
#Visualisation of heat map to see data correlation in sharing_new dataframe 

#Note this visualisation is done before regression just to check the collinearity of different variables 

plt.figure(figsize = (10,7))

sns.heatmap(sharing_new.corr(), annot = True, cmap = 'coolwarm', linecolor = 'white', linewidths=0.1)

# Creating dummies for the categorical columns in the sharing_new dataframe 

sharing_new=pd.get_dummies(sharing_new,drop_first=True)

sharing_new.info()
# We specify random_state so that the train and test data set always have the same rows

np.random.seed(0)

df_train,df_test=train_test_split(sharing_new,train_size=0.70,test_size=0.30,random_state=100)

#verifying the  shape of training dataframe

df_train.shape
#verifying the sample of training dataframe 

df_train.head()
#verifying the columns of Training dataframe 

df_train.info()
#verifying the shape of test dataframe 

df_test.shape
#verifying the sample of test dataframe 

df_test.head()
#verifying the columns of test dataframe 

df_test.info()
#Visualising the correlation after creating dummies in the sharing_new dataframe 

plt.figure(figsize=(25,20))

sns.heatmap(df_train.corr(),annot=True,cmap="YlGnBu")

plt.show()
#Rescaling the features inorder to have comparable scales 

scaler=MinMaxScaler()
#Check the data before scaling 

df_train.head()
#Retrieve the columns in df_train 

df_train.columns 
# Apply scaler() to all the numeric variables

num_vars = ['temp', 'atemp', 'hum', 'windspeed','cnt']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
#Check the data after scaling 

df_train.head()
#check the mean median and other statistics 

df_train.describe()
#Dividing into X and Y sets for the model building

y_train=df_train.pop('cnt')

X_train=df_train
# Running RFE with the output number of the variable equal to 15

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 15)             # running RFE

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
# Running the linear model

lm = sm.OLS(y_train,X_train_rfe).fit()   
# Check the parameters 

lm.params
#Let's see the summary of our linear model

print(lm.summary())
# Check columns of train dataframe 

X_train_rfe.columns
# Calculating the VIF for the model 

# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



#drop const from X_train_refe

X_train_rfe_VIF=X_train_rfe[['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed',

       'season_2', 'season_3', 'season_4', 'mnth_8', 'mnth_9', 'mnth_10',

       'weekday_6', 'weathersit_2', 'weathersit_3']]

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe_VIF.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe_VIF.values, i) for i in range(X_train_rfe_VIF.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#Removing the variable 'season_3' based on its High p-value & High VIF

X_train_new = X_train_rfe_VIF.drop(["season_3"], axis = 1)
# Calculating the VIF for the model 

# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Add a constant

X_train_lm2 = sm.add_constant(X_train_new)



# Create a fitted model

lr2 = sm.OLS(y_train, X_train_lm2).fit()
# Check the parameters obtained



lr2.params
# Print a summary of the linear regression model obtained

print(lr2.summary())
X_train_new = X_train_new.drop(["holiday"], axis = 1)
# Calculating the VIF for the model 

# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Add a constant

X_train_lm3 = sm.add_constant(X_train_new)



# Create a fitted model

lr3 = sm.OLS(y_train, X_train_lm3).fit()
# Check the parameters obtained



lr3.params
# Print a summary of the linear regression model obtained

print(lr3.summary())
X_train_new = X_train_new.drop(["mnth_10"], axis = 1)
# Calculating the VIF for the model 

# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Add a constant

X_train_lm4 = sm.add_constant(X_train_new)



# Create a fitted model

lr4 = sm.OLS(y_train, X_train_lm4).fit()
# Check the parameters obtained



lr4.params
# Print a summary of the linear regression model obtained

print(lr4.summary())
X_train_new = X_train_new.drop(["mnth_8"], axis = 1)
# Calculating the VIF for the model 

# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Add a constant

X_train_lm5 = sm.add_constant(X_train_new)



# Create a fitted model

lr5 = sm.OLS(y_train, X_train_lm5).fit()
# Check the parameters obtained



lr5.params
# Print a summary of the linear regression model obtained

print(lr5.summary())
#Checking the Pvalues

pd.options.display.float_format = '{:.10f}'.format

round(lr5.pvalues,6)
X_train_new = X_train_new.drop(["hum"], axis = 1)
# Calculating the VIF for the model 

# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Add a constant

X_train_lm6 = sm.add_constant(X_train_new)



# Create a fitted model

lr6 = sm.OLS(y_train, X_train_lm6).fit()
# Check the parameters obtained



lr6.params
# Print a summary of the linear regression model obtained

print(lr6.summary())
## Check for p values in the above model 

pd.options.display.float_format = '{:.10f}'.format

round(lr6.pvalues,6)
# Drop working day variable 

X_train_new = X_train_new.drop(["workingday"], axis = 1)
# Calculating the VIF for the model 

# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Add a constant

X_train_lm7 = sm.add_constant(X_train_new)



# Create a fitted model

lr7 = sm.OLS(y_train, X_train_lm7).fit()
# Check the parameters obtained



lr7.params
# Print a summary of the linear regression model obtained

print(lr7.summary())
## Check for p values in the above model 

pd.options.display.float_format = '{:.10f}'.format

round(lr7.pvalues,6)
# Drop working day variable 

X_train_new = X_train_new.drop(["weekday_6"], axis = 1)
# Calculating the VIF for the model 

# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Add a constant

X_train_lm8 = sm.add_constant(X_train_new)



# Create a fitted model

lr8 = sm.OLS(y_train, X_train_lm8).fit()
# Check the parameters obtained



lr8.params
# Print a summary of the linear regression model obtained

print(lr8.summary())
## Check for p values in the above model 

pd.options.display.float_format = '{:.10f}'.format

round(lr8.pvalues,6)
## Residual Analysis of Training Data 

y_train_pred = lr8.predict(X_train_lm8)
res = y_train-y_train_pred



# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((res), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
sharing_lin=sharing_new[[ 'temp', 'atemp', 'hum', 'windspeed','cnt']]



sns.pairplot(sharing_lin, diag_kind='kde')

plt.show()
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Apply scaler() to all numeric variables in test dataset. 



num_vars = ['temp', 'atemp', 'hum', 'windspeed','cnt']



df_test[num_vars] = scaler.transform(df_test[num_vars])
# Checking the sample data from the test dataframe 

df_test.head()
# Check for the mean meadian and the other stats from the test dataframe 

df_test.describe()
#Dividing into X_test and y_test

y_test = df_test.pop('cnt')

X_test = df_test

X_test.info()
#Check shape 

X_test.shape
X_test.columns
#Selecting the variables that were part of final model.

col1=X_train_new.columns

col1
# Now let's use our model to make predictions.



# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[col1]



# Adding a constant variable 

X_test_lm8 = sm.add_constant(X_test_new)
#Check the info of X_tes_lm6

X_test_lm8.info()
# Making predictions using the final model (lr6)



y_pred = lr8.predict(X_test_lm8)
# Plotting y_test and y_pred to understand the linearity 



fig = plt.figure()

plt.scatter(y_test, y_pred, alpha=.5)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16) 
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)


# We already have the value of R^2 (calculated in above step)



rsq_pred=0.7906228342366496
# Get the shape of X_test



X_test.shape
# n is number of rows in X



n = X_test.shape[0]





# Number of features (predictors, p) is the shape along axis 1

p = X_test.shape[1]



# We find the Adjusted R-squared using the formula



adjusted_r2_pred = 1-(1-r2)*(n-1)/(n-p-1)

adjusted_r2_pred
#comparing R Square and Adjusted R square from the final train and predicted model(test model)

rsq_train= lr8.rsquared

rsq_adj_train=lr8.rsquared_adj

print("rsq_train         :",rsq_train)

print("rsq_adj_train     :",rsq_adj_train)

print("rsq_pred          :",rsq_pred)

print("adjusted_r2_pred  :",adjusted_r2_pred)
