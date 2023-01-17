# Supress Warnings
import warnings
warnings.filterwarnings('ignore')
# Importing all required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.metrics import r2_score
#Importing dataset
bike = pd.read_csv('../input/bike-sharing-dataset/Bike_Sharing_BoomBikes_dataset.csv')
bike.head()
# Check the descriptive Information in the dataset
bike.info()
#Check the columns in the datset
bike.columns
# Check descriptive statistics in the dataset
bike.describe()
# Check the Shape of the dataset
bike.shape
# Missing values check in the columns
round(100*bike.isnull().sum()/len(bike.index))
# Missing values check in the rows
round(100* bike.isnull().sum(axis=1)/len(bike.index))
# Create a dummy dataframe (copy of original bike df) for duplicate check
bike_dp = bike

# Checking for duplicates and dropping the entire duplicate row if any
bike_dp.drop_duplicates(subset=None, inplace=True)

bike_dp.shape
# Checking the columns information before drop
bike.columns
# Droping the unused features
bike_1 = bike[['season', 'yr', 'mnth', 'holiday', 'weekday','workingday', 'weathersit', 
               'temp', 'hum', 'windspeed','cnt']]
bike_1.info()
# identify categorical variables
cat_vars = ['season','mnth','holiday','weekday', 'workingday','weathersit']

# identify numeric variables
num_vars = ['temp', 'hum','windspeed','cnt']
# convert dtype of categorical variables
bike_1[cat_vars] = bike_1[cat_vars].astype('category')
# get insights of numeric variable
bike_1.describe()
# get the insights of categorical variables
bike_1.describe(include=['category'])
# maped the season column according to descripttions
bike_1['season'] = bike_1['season'].map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})

# maped the weekday column according to descriptin
bike_1['weekday'] = bike_1['weekday'].map({0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'})


# maped mnth column values (1 to 12 ) as (jan to dec) respectively
bike_1['mnth'] = bike_1['mnth'].map({1:'jan', 2:'feb', 3:'mar', 4:'apr', 5: 'may', 6: 'jun', 7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct',
                             11: 'nov', 12:'dec'})

#  maped weathersit column
bike_1['weathersit'] = bike_1['weathersit'].map({1: 'Clear_FewClouds', 2: 'Mist_Cloudy', 3: 'LightSnow_LightRain', 4: 'HeavyRain_IcePallets'})
# Columns information
bike_1.columns
# Check the data info before proceeding for analysis
bike_1.info()
# Visualizise the pattern of the target variable (demand) over period fo two years
plt.figure(figsize=(15,5))
plt.plot(bike_1.cnt)
plt.show()
# selecting numerical variables

# Box plot
col = 2
row = len(num_vars)//col+1

plt.figure(figsize=(12,8))
plt.rc('font', size=12)
for i in list(enumerate(num_vars)):
    plt.subplot(row, col, i[0]+1)
    sns.boxplot(bike_1[i[1]])    
plt.tight_layout()   
plt.show()
##checking the pie chart distribution of categorical variables
bike_piplot=bike_1[cat_vars]
plt.figure(figsize=(15,15))
plt.suptitle('pie distribution of categorical features', fontsize=20)
for i in range(1,bike_piplot.shape[1]+1):
    plt.subplot(3,3,i)
    f=plt.gca()
    f.set_title(bike_piplot.columns.values[i-1])
    values=bike_piplot.iloc[:,i-1].value_counts(normalize=True).values
    index=bike_piplot.iloc[:,i-1].value_counts(normalize=True).index
    plt.pie(values,labels=index,autopct='%1.1f%%')
#plt.tight_layout()
plt.show()
# Creating a new dataframe of numerical variables
bike_num=bike_1[[ 'temp', 'hum', 'windspeed','cnt']]

sns.pairplot(bike_num, diag_kind='kde')
plt.show()
# checking dataset information 
bike_1.info()
# Build boxplot of all categorical variables (before creating dummies) againt the target variable 'cnt' 
# to see how each of the predictor variable stackup against the target variable.

plt.figure(figsize=(15, 8))
plt.subplot(2,3,1)
sns.boxplot(x = 'season', y = 'cnt', data = bike_1)
plt.subplot(2,3,2)
sns.boxplot(x = 'mnth', y = 'cnt', data = bike_1)
plt.subplot(2,3,3)
sns.boxplot(x = 'weathersit', y = 'cnt', data = bike_1)
plt.subplot(2,3,4)
sns.boxplot(x = 'holiday', y = 'cnt', data = bike_1)
plt.subplot(2,3,5)
sns.boxplot(x = 'weekday', y = 'cnt', data = bike_1)
plt.subplot(2,3,6)
sns.boxplot(x = 'workingday', y = 'cnt', data = bike_1)
#plt.tight_layout(pad=1)
plt.show()
# Check the correlation coefficients to see which variables are highly correlated.
# (im considering only new dataframe variables (bike_1) that were chosen for analysis)

sns.heatmap(bike_1.corr(), annot = True, cmap="YlGnBu")
plt.show()
# Check the datatypes before conversion of the variables
bike_1.info()
# 1. Creating dummy variables
# 2. Drop variable variables for which dummy was created
# 3. Drop first dummy variable for each set of dummies created

bike_1 = pd.get_dummies(bike_1, drop_first = True)
bike_1.info()
# Shape of the new data set
bike_1.shape
# We specify 'random_state' so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(bike_1, train_size = 0.70, test_size = 0.30, random_state = 333)
# Checking the train dataset information
df_train.info()
# Checking the train dataset shape
df_train.shape
# Checking the test dataset info
df_test.info()
# Checking the test dataset shape
df_test.shape
# instantiate and object
scaler = MinMaxScaler()
#fit(): learns xmin, xmax
#transform: (x-xmin)/(xmax - xmin)
#fit_transform ()
# Check vlaues before scaling
df_train.head()
# Checking Columns for infomration
df_train.columns
# Apply scaler() to numerical variables
num_vars = ['temp','hum', 'windspeed','cnt']

# Fit on data
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
# After scaling checking descriptive statistics in the dataset
df_train.describe()
# Check vlaues before scaling
df_test.head()
# Checking Columns for infomration
df_test.columns
# Apply scaler() to all numeric variables in test dataset. Note: we will only use scaler.transform, 
# as we want to use the metrics that the model learned from the training data to be applied on the test data. 
# In other words, we want to prevent the information leak from train to test dataset.

num_vars = ['temp', 'hum', 'windspeed','cnt']

df_test[num_vars] = scaler.transform(df_test[num_vars])
# Check the test dataset
df_test.head()
# Check the test dataset statistics
df_test.describe()
# Creating X and y data dataframe for train set
y_train = df_train.pop('cnt')
X_train =df_train
X_train.head()
# checking the y_train Data
y_train.head()
# Creating X and y data dataframe for test set
y_test = df_test.pop('cnt')
X_test =df_test
X_test.head()
# checking the y_train Data
y_test.head()
# Running RFE with the output number of the variable equal to 15
lm = LinearRegression()
lm.fit(X_train, y_train)

# running RFE
rfe = RFE(lm, 15)             
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# Select columns
col = X_train.columns[rfe.support_]
col
# Columns
X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]
# Add a constant
X_train_lm1 = sm.add_constant(X_train_rfe)

# Create a first fitted model
lr1 = sm.OLS(y_train, X_train_lm1).fit()

#Check parameters 
lr1.params
# Summary of the linear regression model
print(lr1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_rfe.drop(["mnth_may"], axis = 1)
# Add a constant
X_train_lm2 = sm.add_constant(X_train_new)

# Create a fitted model
lr2 = sm.OLS(y_train, X_train_lm2).fit()

# Check parameters
lr2.params
# Summary of the linear regression model
print(lr2.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_new.drop(["hum"], axis = 1)
# Add a constant
X_train_lm3 = sm.add_constant(X_train_new)

# Create a fitted model
lr3 = sm.OLS(y_train, X_train_lm3).fit()

# Check parameters
lr3.params
# Summary of the linear regression model
print(lr3.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_new.drop(["season_fall"], axis = 1)
# Add a constant
X_train_lm4 = sm.add_constant(X_train_new)

# Create a fitted model
lr4 = sm.OLS(y_train, X_train_lm4).fit()

# Check parameters
lr4.params
# Summary of the linear regression model
print(lr4.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_new.drop(["mnth_oct"], axis = 1)
# Add a constant
X_train_lm5 = sm.add_constant(X_train_new)

# Create a fitted model
lr5 = sm.OLS(y_train, X_train_lm5).fit()

# Check parameters
lr5.params
# Summary of the linear regression model
print(lr5.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_new.drop(["mnth_mar"], axis = 1)
# Add a constant
X_train_lm6 = sm.add_constant(X_train_new)

# Create a fitted model
lr6 = sm.OLS(y_train, X_train_lm6).fit()

# Check parameters
lr6.params
# Summary of the linear regression model
print(lr6.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# List down model varibales and its coefficients

# assign final model to lm_final
lm_final = lr6

# list down and check variables of final model
var_final = list(lm_final.params.index)
var_final.remove('const')
print('Final Selected Variables:', var_final)

# Print the coefficents of final varible
print('\033[1m{:10s}\033[0m'.format('\nCoefficent for the variables are:'))
print(round(lm_final.params,3))
# predict traint dataset
y_train_pred = lr6.predict(X_train_lm6)
# Plot the histogram of the error terms
res_train = y_train - y_train_pred

fig = plt.figure()
sns.distplot((res_train), bins = 20)
# Plot heading
fig.suptitle('Error Terms Train', fontsize = 20)                   
# X-label
plt.xlabel('Errors', fontsize = 18)                         
# Plotting y_test and y_pred to understand the spread

fig = plt.figure(figsize = (8,5))
plt.scatter(y_train, res_train, alpha=.6)

# Plot heading 
fig.suptitle('Error Terms Train', fontsize = 20)             

# label
plt.xlabel('y_train_pred', fontsize = 18)                          
plt.ylabel('Residual', fontsize = 16)
#Selecting the variables that were part of final model.
col1=X_train_new.columns

X_test=X_test[col1]

# Adding constant variable to test dataframe
X_test_lm6 = sm.add_constant(X_test)

X_test_lm6.info()
# predict test dataset
y_test_pred = lr6.predict(X_test_lm6)
# Plot the histogram of the error terms
res_test = y_test-y_test_pred
plt.title('Error Terms Test', fontsize=16) 
sns.distplot(res_test)
plt.show()
# Get R-Squared fro test dataset
r2_test = r2_score(y_test, y_pred = y_test_pred)
print('r2_test: ', round(r2_test,3))
# We already have the value of R^2 (calculated in above step)
r2=r2_test
# Get the shape of X_test

X_test.shape
# n is number of rows in X
n = X_test.shape[0]

# Number of features (predictors, p) is the shape along axis 1
p = X_test.shape[1]

# We find the Adjusted R-squared using the formula
r2_test_adj = 1-(1-r2)*(n-1)/(n-p-1)
print('r2_test_adj:', round(r2_test_adj,3))
# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, res_test, alpha=.5)
fig.suptitle('Error Terms Test', fontsize = 20)             # Plot heading 
plt.xlabel('y_test_pred', fontsize = 18)                          # X-label
plt.ylabel('Residual', fontsize = 16)
# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_test_pred)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_test_pred', fontsize = 16)      
# Print R Squared and adj. R Squared
print('R- Sqaured train: ', round(lm_final.rsquared,3), '  Adj. R-Squared train:', round(lm_final.rsquared_adj,3) )
print('R- Sqaured test : ', round(r2_test,2), '  Adj. R-Squared test :', round(r2_test_adj,3))
# Print the coefficents of final varible
print('\033[1m{:10s}\033[0m'.format('\nCoefficent for the variables are:'))
print(round(lm_final.params,3))