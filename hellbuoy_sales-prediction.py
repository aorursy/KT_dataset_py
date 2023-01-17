# import all libraries and dependencies for dataframe

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta



# import all libraries and dependencies for data visualization

pd.options.display.float_format='{:.4f}'.format

plt.rcParams['figure.figsize'] = [8,8]

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', -1) 

sns.set(style='darkgrid')

import matplotlib.ticker as ticker

import matplotlib.ticker as plticker



# import all libraries and dependencies for machine learning

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.base import TransformerMixin

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import r2_score
# Reading the advertising company file on which analysis needs to be done

df_sales = pd.read_csv('../input/sales-advertisment/advertising.csv')
df_sales.head()
df_sales.shape
df_sales.info()
df_sales.describe()
# Calculating the Missing Values % contribution in DF



df_null = df_sales.isna().mean().round(4) * 100



df_null.sort_values(ascending=False).head()
df_sales.dtypes
# Outlier Analysis of target variable with maximum amount of Inconsistency



outliers = ['Sales']

plt.rcParams['figure.figsize'] = [8,8]

sns.boxplot(data=df_sales[outliers], orient="v", palette="Set1" ,whis=1.5,saturation=1, width=0.7)

plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')

plt.ylabel("Sales Range", fontweight = 'bold')

plt.xlabel("Continuous Variable", fontweight = 'bold')
# checking for duplicates



df_sales.loc[df_sales.duplicated()]
plt.figure(figsize=(8,8))



plt.title('Sales Distribution Plot')

sns.distplot(df_sales['Sales'])
sns.pairplot(df_sales)
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (10, 8))

df_corr = df_sales.corr()

ax = sns.heatmap(df_corr, annot=True, cmap="RdYlGn") 

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
# We specify this so that the train and test data set always have the same rows, respectively

# We divide the df into 70/30 ratio



np.random.seed(0)

df_train, df_test = train_test_split(df_sales, train_size = 0.7, test_size = 0.3, random_state = 100)
df_train.head()
cols = ['TV','Radio','Newspaper']
# Scatter Plot of independent variables vs dependent variables



fig,axes = plt.subplots(1,3,figsize=(20,6))

for seg,col in enumerate(cols):

    x,y = seg//3,seg%3

    an=sns.scatterplot(x=col, y='Sales' ,data=df_sales, ax=axes[y])

    plt.setp(an.get_xticklabels(), rotation=45)

   

plt.subplots_adjust(hspace=0.5)
y_train = df_train.pop('Sales')

X_train = df_train
X_train_1 = X_train['TV']
# Add a constant

X_train_1c = sm.add_constant(X_train_1)



# Create a first fitted model

lr_1 = sm.OLS(y_train, X_train_1c).fit()
# Check parameters created



lr_1.params
# Let's visualise the data with a scatter plot and the fitted regression line



plt.scatter(X_train_1c.iloc[:, 1], y_train)

plt.plot(X_train_1c.iloc[:, 1], 6.9487 + 0.0545*X_train_1c.iloc[:, 1], 'r')

plt.show()
# Print a summary of the linear regression model obtained

print(lr_1.summary())
X_train_2 = X_train[['TV', 'Radio']]
# Add a constant

X_train_2c = sm.add_constant(X_train_2)



# Create a second fitted model

lr_2 = sm.OLS(y_train, X_train_2c).fit()
lr_2.params
print(lr_2.summary())
X_train_3 = X_train[['TV', 'Radio', 'Newspaper']]
# Add a constant

X_train_3c = sm.add_constant(X_train_3)



# Create a third fitted model

lr_3 = sm.OLS(y_train, X_train_3c).fit()
lr_3.params
print(lr_3.summary())
# Predicting the price of training set.

y_train_sales = lr_2.predict(X_train_2c)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_sales), bins = 20)

fig.suptitle('Error Terms Analysis', fontsize = 20)                   

plt.xlabel('Errors', fontsize = 18)
y_test = df_test.pop('Sales')

X_test = df_test
# Adding constant

X_test_1 = sm.add_constant(X_test)



X_test_new = X_test_1[X_train_2c.columns]
# Making predictions using the final model

y_pred = lr_2.predict(X_test_new)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)   

plt.xlabel('y_test ', fontsize=18)                       

plt.ylabel('y_pred', fontsize=16)    
r2_score(y_test, y_pred)