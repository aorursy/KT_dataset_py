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
import pandas as pd    
dataset = pd.read_csv('../input/fastract-wrist-watch-features-and-price/dataset.csv')
df = pd.DataFrame(dataset)
df.head()
df['price'] = df['price'].astype(str)
df['price'] = df['price'].str.replace(',', '').str.replace('₹', '').astype(int)
df['price'].dtype
df['price'].describe()
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")
ax = sns.boxplot(x=df["price"])
Q1 =  df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

#removing outliers from data bsed on price
outliers = []
for i in df.index :
    if df['price'][i] > Q1+1.5*IQR or df['price'][i] < Q1 -  1.5*IQR:
        outliers.append(i)
        
df.drop(index = outliers,inplace = True)
sns.set(style="whitegrid")
ax = sns.boxplot(x=df["price"])
df.dtypes
df['thickness'] = df['thickness'].astype(str)
df['thickness'] = df['thickness'].str.replace(' ', '').str.replace('mm', '').astype(float)

df['weight'] = df['weight'].astype(str)
df['weight'] = df['weight'].str.replace(' ', '').str.replace('g', '').astype(float)

df['width'] = df['width'].astype(str)
df['width'] = df['width'].str.replace(' ', '').str.replace('mm', '').str.replace('cm', '').astype(float)

df['Diameter'] = df['Diameter'].astype(str)
df['Diameter'] = df['Diameter'].str.replace(' ', '').str.replace('mm', '').astype(float)

df.dtypes
df['water_resistant'].unique()
Numeric = ['weight','width','thickness','Diameter','price']
categorical = ['water_resistant','Display_type','strap_material','box_material','occasion','series','strap_type']
num_df = df.loc[: , Numeric]
cat_df = df.loc[: , categorical]
import matplotlib.pyplot as plt
plt.title('wrist watch Price Distribution Plot')
sns.distplot(df['price'])
ax = sns.pairplot(num_df)
corr=num_df.corr()
corr.style.background_gradient(cmap="inferno")
plt.figure(figsize=(20, 15))
plt.subplot(3,3,1)
sns.boxplot(x = 'water_resistant', y = 'price', data = df)
plt.subplot(3,3,2)
sns.boxplot(x = 'Display_type', y = 'price', data = df)
plt.figure(figsize=(20, 10))
sns.boxplot(x = 'strap_material', y = 'price', data = df)
df['strap_material'].unique()
full_cat = ['Silicone Strap', 'Stainless Steel Strap', 'Leather Strap',
       'Metal Strap', 'Silicon Strap', 'Rubber Strap', 'Sillicone Strap',
       'Resin Strap', 'Tan Strap', 'Genuine Leather Strap', 'metal Strap','Denim Strap', 'Plastic Strap',
       'Polyurethane Strap', 'LEATHER Strap', 'Steel Strap',
       'Fabric Strap']
for i in df.index:
    if df['strap_material'][i] in full_cat:
        df['strap_material'][i] = 'cat1'
df['strap_material'].unique()
plt.figure(figsize=(20, 10))
sns.boxplot(x = 'strap_material', y = 'price', data = df)
df['box_material'].unique()
plt.figure(figsize=(20, 10))
sns.boxplot(x = 'box_material', y = 'price', data = df)
df['box_material'].unique()
full_cat2 = ['Cardboard', 'Metal', 'Plastic', 'Brass Case','Plastic Box',]
for i in df.index:
    if df['box_material'][i] in full_cat2:
        df['box_material'][i] = 'cat1'
plt.figure(figsize=(20, 10))
sns.boxplot(x = 'box_material', y = 'price', data = df)
df['occasion'].unique()
plt.figure(figsize=(20, 10))
sns.boxplot(x = 'occasion', y = 'price', data = df)
df['occasion'].value_counts().plot(kind='bar')
plt.xlabel('occasion')
plt.ylabel('Frequency')
plt.show()
df['series'].unique()
df['series'].value_counts().plot(kind = 'bar')
plt.figure(figsize=(20, 10))
sns.boxplot(x = 'series', y = 'price', data = df)
df['strap_type'].unique()
plt.figure(figsize=(20, 10))
sns.boxplot(x = 'strap_type', y = 'price', data = df)
df['strap_type'].value_counts().plot(kind = 'bar')
print(cat_df)
dummy_cat = ['water_resistant','Display_type','strap_material','box_material','occasion','series','strap_type']
dummies = pd.get_dummies(df[dummy_cat])
dummies.shape
dummies = pd.get_dummies(df[dummy_cat], drop_first = True)
dummies.shape
# Drop the original cat variables as dummies are already created

df.drop( dummy_cat, axis = 1, inplace = True)
df.shape
df = df.drop(columns = ['Diameter','width'])
df.head()
corr=df.corr()
corr.style.background_gradient(cmap="inferno")
df = pd.concat([df, dummies], axis = 1)
df.head()
df.shape

df.head()
# We specify this so that the train and test data set always have the same rows, respectively
# We divide the df into 70/30 ratio

from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train, df_test = train_test_split(df, train_size = 0.8, test_size = 0.2, random_state = 100)

scaler = preprocessing.StandardScaler()
sig_num_col = ['price','thickness','weight']
# Apply scaler() to all the columns except the 'dummy' variables
import warnings
warnings.filterwarnings("ignore")

df_train[sig_num_col] = scaler.fit_transform(df_train[sig_num_col])
df = df.drop(columns = 'water_resistant')
df_test.head()
df_train.head()
# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (20, 20))
sns.heatmap(df_train.corr(), cmap="RdYlGn")
plt.show()
y_train = df_train.pop('price')
X_train = df_train
y_train.shape
X_train_1 = X_train['thickness']
# Add a constant
X_train_1c = sm.add_constant(X_train_1)

# Create a first fitted model
lr_1 = sm.OLS(y_train, X_train_1c).fit()
lr_1.params
# Let's visualise the data with a scatter plot and the fitted regression line

plt.scatter(X_train_1c.iloc[:, 1], y_train)
plt.plot(X_train_1c.iloc[:, 1], 0.8062*X_train_1c.iloc[:, 1], 'r')
plt.show()
# Print a summary of the linear regression model obtained
print(lr_1.summary())
X_train_2 = X_train[['thickness', 'weight']]
# Add a constant
X_train_2c = sm.add_constant(X_train_2)

# Create a second fitted model
lr_2 = sm.OLS(y_train, X_train_2c).fit()
lr_2.params
print(lr_2.summary())
# Running RFE with the output number of the variable equal to 15
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 15)             
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# Selecting the variables which are in support

col_sup = X_train.columns[rfe.support_]
col_sup
# Creating X_train dataframe with RFE selected variables

X_train_rfe = X_train[col_sup]
# Adding a constant variable and Build a first fitted model
import statsmodels.api as sm  
X_train_rfec = sm.add_constant(X_train_rfe)
lm_rfe = sm.OLS(y_train,X_train_rfec).fit()

#Summary of linear model
print(lm_rfe.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Dropping highly correlated variables and insignificant variables

X_train_rfe1 = X_train_rfe.drop('strap_type_Genuine leather', 1,)

# Adding a constant variable and Build a second fitted model

X_train_rfe1c = sm.add_constant(X_train_rfe1)
lm_rfe1 = sm.OLS(y_train, X_train_rfe1c).fit()

#Summary of linear model
print(lm_rfe1.summary())
# Dropping highly correlated variables and insignificant variables

X_train_rfe2 = X_train_rfe1.drop('series_loopholes', 1,)

# Adding a constant variable and Build a third fitted model

X_train_rfe2c = sm.add_constant(X_train_rfe2)
lm_rfe2 = sm.OLS(y_train, X_train_rfe2c).fit()

#Summary of linear model
print(lm_rfe2.summary())
# Dropping highly correlated variables and insignificant variables

X_train_rfe3 = X_train_rfe2.drop('series_Tattoo', 1,)

# Adding a constant variable and Build a fourth fitted model
X_train_rfe3c = sm.add_constant(X_train_rfe3)
lm_rfe3 = sm.OLS(y_train, X_train_rfe3c).fit()

#Summary of linear model
print(lm_rfe3.summary())
# Dropping highly correlated variables and insignificant variables

X_train_rfe4 = X_train_rfe3.drop('series_Upgrades', 1,)

# Adding a constant variable and Build a fifth fitted model
X_train_rfe4c = sm.add_constant(X_train_rfe4)
lm_rfe4 = sm.OLS(y_train, X_train_rfe4c).fit()

#Summary of linear model
print(lm_rfe4.summary())
# Dropping highly correlated variables and insignificant variables

X_train_rfe5 = X_train_rfe4.drop('series_Bikers', 1,)

# Adding a constant variable and Build a sixth fitted model
X_train_rfe5c = sm.add_constant(X_train_rfe5)
lm_rfe5 = sm.OLS(y_train, X_train_rfe5c).fit()

#Summary of linear model
print(lm_rfe5.summary())
# Dropping highly correlated variables and insignificant variables

X_train_rfe6 = X_train_rfe5.drop('series_Fundamentals', 1,)

# Adding a constant variable and Build a sixth fitted model
X_train_rfe6c = sm.add_constant(X_train_rfe6)
lm_rfe6 = sm.OLS(y_train, X_train_rfe6c).fit()

#Summary of linear model
print(lm_rfe6.summary())
# Dropping highly correlated variables and insignificant variables

X_train_rfe7 = X_train_rfe6.drop('series_Extreme Hybrid', 1,)

# Adding a constant variable and Build a sixth fitted model
X_train_rfe7c = sm.add_constant(X_train_rfe7)
lm_rfe7 = sm.OLS(y_train, X_train_rfe7c).fit()

#Summary of linear model
print(lm_rfe7.summary())
# Dropping highly correlated variables and insignificant variables

X_train_rfe8 = X_train_rfe7.drop('series_Tees Valentines', 1,)

# Adding a constant variable and Build a sixth fitted model
X_train_rfe8c = sm.add_constant(X_train_rfe8)
lm_rfe8 = sm.OLS(y_train, X_train_rfe8c).fit()

#Summary of linear model
print(lm_rfe8.summary())
# Dropping highly correlated variables and insignificant variables

X_train_rfe9 = X_train_rfe8.drop('series_Star Wars', 1,)

# Adding a constant variable and Build a sixth fitted model
X_train_rfe9c = sm.add_constant(X_train_rfe9)
lm_rfe9 = sm.OLS(y_train, X_train_rfe9c).fit()

#Summary of linear model
print(lm_rfe9.summary())
# Dropping highly correlated variables and insignificant variables

X_train_rfe10 = X_train_rfe9.drop('series_Trendies', 1,)

# Adding a constant variable and Build a sixth fitted model
X_train_rfe10c = sm.add_constant(X_train_rfe10)
lm_rfe10 = sm.OLS(y_train, X_train_rfe10c).fit()

#Summary of linear model
print(lm_rfe10.summary())
# Dropping highly correlated variables and insignificant variables

X_train_rfe11 = X_train_rfe10.drop('series_Party', 1,)

# Adding a constant variable and Build a sixth fitted model
X_train_rfe11c = sm.add_constant(X_train_rfe11)
lm_rfe11 = sm.OLS(y_train, X_train_rfe11c).fit()

#Summary of linear model
print(lm_rfe11.summary())
# Predicting the price of training set.
y_train_price = lm_rfe11.predict(X_train_rfe11c)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms Analysis', fontsize = 20)                   
plt.xlabel('Errors', fontsize = 18)
import warnings
warnings.filterwarnings("ignore")

df_test[sig_num_col] = scaler.transform(df_test[sig_num_col])
df_test.shape
y_test = df_test.pop('price')
X_test = df_test
# Adding constant
X_test_1 = sm.add_constant(X_test)

X_test_new = X_test_1[X_train_rfe10c.columns]
# Making predictions using the final model
y_pred = lm_rfe10.predict(X_test_new)
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)   
plt.xlabel('y_test ', fontsize=18)                       
plt.ylabel('y_pred', fontsize=16)  