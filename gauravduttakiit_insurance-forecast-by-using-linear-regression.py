import warnings

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
insurance= pd.read_csv("/kaggle/input/insurance/insurance.csv")
insurance.head()
insurance.info()
insurance.describe()
insurance.shape
# percentage of missing values in each column

round(100*(insurance.isnull().sum()/len(insurance)),2).sort_values(ascending = False)
# percentage of missing values in each row

round(100*(insurance.isnull().sum(axis=1)/len(insurance)),2).sort_values(ascending = False)[:5]
insurance_dub=insurance.copy()

# Checking for duplicates and dropping the entire duplicate row if any

insurance_dub.drop_duplicates(subset=None, inplace=True)
insurance_dub.shape
insurance.shape
for col in insurance:

    print(insurance[col].value_counts(ascending=False), '\n\n\n')
# Check the datatypes before convertion

insurance.info()
# Convert to 'category' data type



insurance['sex']=insurance['sex'].astype('category')

insurance['smoker']=insurance['smoker'].astype('category')

insurance['children']=insurance['children'].astype('category')

insurance['region']=insurance['region'].astype('category')
# This code does 3 things:

# 1) Create Dummy variable

# 2) Drop original variable for which the dummy was created

# 3) Drop first dummy variable for each set of dummies created.



insurance_new = pd.get_dummies(insurance, drop_first=True)

insurance_new.info()
insurance_new.shape


# Check the shape before spliting



insurance_new.shape
# Check the info before spliting



insurance_new.info()
from sklearn.model_selection import train_test_split



# We should specify 'random_state' so that the train and test data set always have the same rows, respectively



np.random.seed(0)

df_train, df_test = train_test_split(insurance_new, train_size = 0.70, test_size = 0.30, random_state = 100)
df_train.info()
df_train.shape
df_test.info()
df_test.shape
df_train.info()
df_train.columns
# Create a new dataframe of only numeric variables:



insurance_num=df_train[[ 'age', 'bmi', 'charges']]



sns.pairplot(insurance_num, diag_kind='kde')

plt.show()
df_train.info()
# Build boxplot of all categorical variables (before creating dummies) againt the target variable 'cnt' 

# to see how each of the predictor variable stackup against the target variable.



plt.figure(figsize=(25, 10))

plt.subplot(2,2,1)

sns.boxplot(x = 'sex', y = 'charges', data = insurance)

plt.subplot(2,2,2)

sns.boxplot(x = 'children', y = 'charges', data = insurance)

plt.subplot(2,2,3)

sns.boxplot(x = 'smoker', y = 'charges', data = insurance)

plt.subplot(2,2,4)

sns.boxplot(x = 'region', y = 'charges', data = insurance)

plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated. Note:

# here we are considering only those variables (dataframe: insurance_new) that were chosen for analysis



plt.figure(figsize = (25,10))

sns.heatmap(insurance_new.corr(), annot = True, cmap="RdBu")

plt.show()
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
df_train.head()
df_train.columns
# Apply scaler() to all the numeric variables



num_vars = ['age', 'bmi', 'charges']



df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
# Checking values after scaling

df_train.head()
df_train.describe()
y_train = df_train.pop('charges')

X_train = df_train
#Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 6

lm= LinearRegression()

lm.fit(X_train,y_train)

rfe= RFE(lm,6)

rfe=rfe.fit(X_train,y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col=X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]

# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
import statsmodels.api as sm



# Add a constant

X_train_lm1 = sm.add_constant(X_train_rfe)



# Create a first fitted model

lr1 = sm.OLS(y_train, X_train_lm1).fit()
# Check the parameters obtained



lr1.params
# Print a summary of the linear regression model obtained

print(lr1.summary())
X_train_new = X_train_rfe.drop(["children_5"], axis = 1)
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



# Create a first fitted model

lr2 = sm.OLS(y_train, X_train_lm2).fit()
# Check the parameters obtained



lr2.params
# Print a summary of the linear regression model obtained

print(lr2.summary())
X_train_new = X_train_new.drop(["children_4"], axis = 1)
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



# Create a first fitted model

lr3 = sm.OLS(y_train, X_train_lm3).fit()
# Check the parameters obtained



lr3.params
# Print a summary of the linear regression model obtained

print(lr3.summary())
X_train_new = X_train_new.drop(["children_2"], axis = 1)
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



# Create a first fitted model

lr4 = sm.OLS(y_train, X_train_lm4).fit()
# Check the parameters obtained



lr4.params
# Print a summary of the linear regression model obtained

print(lr4.summary())
y_train_pred = lr4.predict(X_train_lm4)
res = y_train-y_train_pred

# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((res), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)  
insurance_new=insurance_new[[ 'bmi', 'age', 'charges']]



sns.pairplot(insurance_new, diag_kind='kde')

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
# Apply scaler() to all numeric variables in test dataset. Note: we will only use scaler.transform, 

# as we want to use the metrics that the model learned from the training data to be applied on the test data. 

# In other words, we want to prevent the information leak from train to test dataset.



num_vars = ['age', 'bmi', 'charges']



df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.head()
df_test.describe()
y_test = df_test.pop('charges')

X_test = df_test

X_test.info()
#Selecting the variables that were part of final model.

col1=X_train_new.columns

X_test=X_test[col1]

# Adding constant variable to test dataframe

X_test_lm4 = sm.add_constant(X_test)

X_test_lm4.info()
# Making predictions using the final model (lr4)



y_pred = lr4.predict(X_test_lm4)

# Plotting y_test and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y_test, y_pred, alpha=.5)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16) 

plt.show()
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
# We already have the value of R^2 (calculated in above step)



r2=0.75783003115855
# Get the shape of X_test

X_test.shape
# n is number of rows in X



n = X_test.shape[0]





# Number of features (predictors, p) is the shape along axis 1

p = X_test.shape[1]



# We find the Adjusted R-squared using the formula



adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)

adjusted_r2