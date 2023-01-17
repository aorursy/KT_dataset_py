'''importing the required libraries
'''
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import seaborn as sns
import matplotlib.pyplot as plt

# Supress Warnings

import warnings
warnings.filterwarnings('ignore')
'''
    Please update the location of the CSV file.
    reading the dataset from the required location
'''
df = pd.read_csv(r'../input/carprice-assignment/CarPrice_Assignment.csv')
df.describe()
df.info()
'''
    splitting the column name CarName to both carname and company_name
'''
df[['company_name','CarName']] = df.CarName.apply(lambda x: pd.Series(str(x).split(" ",1)))
df.info()
'''
    checking for the data quality in the column CarName
'''
df.CarName.unique()
'''
    checking for the data quality in the company_name
'''

df.company_name.unique()
'''
    replacing the column vaues to correct the typing mistakes to resolve the data quality issues
'''

df['company_name'] = df['company_name'].replace('maxda', 'mazda')
df['company_name'] = df['company_name'].replace('Nissan', 'nissan')
df['company_name'] = df['company_name'].replace('porcshce', 'porsche')
df['company_name'] = df['company_name'].replace('toyouta', 'toyota')
df['company_name'] = df['company_name'].replace('vokswagen', 'volkswagen')
df['company_name'] = df['company_name'].replace('vw', 'volkswagen')
'''verifying that the data quakity issues are no longer present in the data set'''
df.company_name.unique()
     #start visualising
sns.pairplot(df)
plt.show()

# we should go with linear regresssion because for few variables we can see a 
#positive co-relation between the numerical variables
#in order to visualise a categorical variable we should use a box plot
plt.figure(figsize=(30, 18))

plt.subplot(3, 4, 1)
sns.boxplot(x = 'enginetype', y = 'price', data = df)

plt.subplot(3, 4, 2)
sns.boxplot(x = 'fueltype', y = 'price', data = df)

plt.subplot(3, 4, 3)
sns.boxplot(x = 'aspiration', y = 'price', data = df)

plt.subplot(3, 4, 4)
sns.boxplot(x = 'doornumber', y = 'price', data = df)

plt.subplot(3, 4, 5)
sns.boxplot(x = 'carbody', y = 'price', data = df)

plt.subplot(3, 4, 6)
sns.boxplot(x = 'drivewheel', y = 'price', data = df)

plt.subplot(3, 4, 7)
sns.boxplot(x = 'carbody', y = 'price', data = df)

plt.subplot(3, 4, 8)
sns.boxplot(x = 'cylindernumber', y = 'price', data = df)

plt.subplot(3, 4, 9)
sns.boxplot(x = 'fuelsystem', y = 'price', data = df)

plt.subplot(3, 4, 10)
sns.boxplot(x = 'symboling', y = 'price', data = df)
#boxplot boundaries represents - 25%, median, 75 %
'''plotting the heatmap to find the correlation amongst the columns'''
plt.figure(figsize=(20,12))
sns.heatmap(df.corr(),annot=True,cmap="YlGnBu")
plt.show()
# creating dummy variables for all the categorical columns

dummy_var = ['carbody','symboling','fuelsystem','cylindernumber','drivewheel','carbody','doornumber','aspiration',
              'fueltype','enginetype','company_name']
dummy_var_df = pd.get_dummies(df[dummy_var],drop_first=True)

dummy_var_df.head()
'''now concat the dummy data frame with a main dataframe'''
df = pd.concat([df,dummy_var_df],axis=1)
df.head()
'''drop the columns for which the dummy variables are already created'''
df = df.drop(dummy_var,axis=1)
df.head()
'''generating the train and test data set'''
df_train, df_test= train_test_split(df,train_size=0.7,random_state=100)
print(df_train.shape)
print(df_test.shape)
df_train.info()
'''convert the ctaegorical column enginelocation to a continuous variable'''
df_train.enginelocation.unique()
df_train.enginelocation.value_counts()
df['enginelocation'] = df['enginelocation'].replace('front', '1')
df['enginelocation'] = df['enginelocation'].replace('rear', '0')
# min -max scaling

# 1. Instantiate the objest of the imported class

scaler = MinMaxScaler()

num_variables =['wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg','price']

#2. Fit on data
df_train[num_variables] = scaler.fit_transform(df_train[num_variables])
df_train.head()
#df_train = df_train[num_variables]
df_train.enginelocation.unique()
y_train = df_train.pop('price')
X_train = df_train
y_train.head()
X_train.pop('CarName')
X_train.pop('enginelocation')
# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 10)             # running RFE
rfe = rfe.fit(X_train, y_train)

X_train.info()
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]
col
X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]
X_train_rfe = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
X_train_new = X_train_rfe.drop(["cylindernumber_three"], axis = 1)
X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print(lm.summary())
X_train_new = X_train_new.drop(['const'], axis=1)
# Calculate the VIFs for the new model
vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_new.drop(['boreratio'], axis=1)
X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print(lm.summary())
'''Dropping the company name company_name_porsche feature as it has the highest p-value'''
X_train_new = X_train_new.drop(['company_name_porsche'], axis=1)
X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print(lm.summary())
'''removing the curbweight feature because of high p-value'''
X_train_new = X_train_new.drop(['curbweight'], axis=1)
vif = pd.DataFrame()
X = X_train_lm
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
'''dropping the feature company_name_subaru as it has very high VIF and this shows multi collinearity'''
X_train_new = X_train_new.drop(['company_name_subaru'], axis=1)
X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print(lm.summary())
'''dropping the feature enginetype_ohcf as it has very high p-value'''
X_train_new = X_train_new.drop(['enginetype_ohcf'], axis=1)
X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print(lm.summary())
vif = pd.DataFrame()
X = X_train_lm
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
y_train_price = lm.predict(X_train_lm)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label
df_test[num_variables] = scaler.transform(df_test[num_variables])
df_test.describe()
y_test = df_test.pop('price')
X_test = df_test
# Now let's use our model to make predictions.

# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)
# Making predictions
y_pred = lm.predict(X_test_new)
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label
# evaluation
r2_score(y_true=y_test, y_pred = y_pred)