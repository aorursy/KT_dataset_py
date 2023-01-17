# Data dicretory file
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

'''Pandas DataFrame is two-dimensional size-mutable, 
potentially heterogeneous tabular data structure with labeled axes (rows and columns). 
A Data frame is a two-dimensional data structure, i.e., data is aligned in a tabular fashion in rows and columns. 
Pandas DataFrame consists of three principal components, the data, rows, and columns.
'''

# Create a Data frame
df = pd.read_csv('../input/googleplaystore1/googleplaystore.csv', header=0)
# Show the first 5 rows of data

df.head()
# df.describe(): summary statistics info of data

df.describe()
# df.boxplot() : A graph of summary statistics info of data

df.boxplot()
# df.info(): to know the shape of data and summary info of each column

df.info()
# calculate null(embty) values 

null_values = df.isnull().sum()
print(null_values)
# Show rating more than 5 to remove

df[df.Rating > 5]
# remove the column

df.drop([10472],inplace=True)

df.hist()
# Fill the null values with appropriate values using aggregate function "median" as the rating histogram is right skewed and it's numerical values

#Define a function impute_median
def impute_median(series):
    return series.fillna(series.median())
df.Rating = df['Rating'].transform(impute_median)
# Fill the null values with appropriate values using aggregate function "mode" for categorical values
df['Type'].fillna(str(df['Type'].mode().values[0]), inplace=True)
df['Current Ver'].fillna(str(df['Current Ver'].mode().values[0]), inplace=True)
df['Android Ver'].fillna(str(df['Android Ver'].mode().values[0]), inplace=True)
# Convert the categorical values into numerical 

df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
df['Price'] = df['Price'].apply(lambda x: float(x))

df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')

df['Installs'] = df['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else str(x))
df['Installs'] = df['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else str(x))
df['Installs'] = df['Installs'].apply(lambda x: float(x))
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(25,10))
plt.xticks(fontsize=20) # X Ticks
plt.yticks(fontsize=20) # Y Ticks
ax.set_title('Number of ratings per category', fontweight="bold", size=20) # Title
ax.set_ylabel('Number of ratings', fontsize = 25) # Y label
ax.set_xlabel('Categories', fontsize = 25) # X label
df.groupby(['Category']).count()['Rating'].plot(ax=ax, kind='bar')
import seaborn as sns

fig, ax = plt.subplots(figsize=(30,10))
sns.boxplot(x='Category', y='Rating', data=df,)
plt.xticks(fontsize=20, rotation=90) # X Ticks
plt.yticks(fontsize=20) # Y Ticks
ax.set_title('Length of ratings per Category', fontweight="bold", size=25) # Title
ax.set_ylabel('Length', fontsize = 25) # Y label
ax.set_xlabel('Category', fontsize = 25) # X label
plt.show()
import numpy as np # linear algebra

df_gp_cat = df.groupby('Category')
y = df_gp_cat['Price'].agg(np.sum)
z = df_gp_cat['Reviews'].agg(np.mean)
plt.figure(figsize=(16,5))
plt.plot(y,'r--', color='k')
plt.xticks(rotation=90)
plt.title('Pricing per Category')
plt.xlabel('Category')
plt.ylabel('Prices')
plt.show()
plt.figure(figsize=(16,5))
plt.plot(z,'bs', color='g')
plt.xticks(rotation=90)
plt.title('Reviews per Category')
plt.xlabel('Categories')
plt.ylabel('Reviews')
plt.show()
# convert all categorical data into numbers

df['Category']= pd.factorize( df['Category'] )[0].astype(int)
df['Type']= pd.factorize( df['Genres'] )[0].astype(int)
df['Genres']= pd.factorize( df['Genres'] )[0].astype(int)
df['Content Rating']= pd.factorize( df['Genres'] )[0].astype(int)
# Split the data

X = df.drop(labels = ['App', 'Size', 'Last Updated', 'Current Ver', 'Android Ver'],axis = 1)
y = df.Rating

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Applying Gradient Boosting Regressor Model 

from sklearn.ensemble import GradientBoostingRegressor
'''
sklearn.ensemble.GradientBoostingRegressor(loss='ls’, learning_rate=0.1,n_estimators=100, subsample=
                                           1.0, criterion='friedman_mse’,min_samples_split=2,min_samples_leaf=1,
                                           min_weight_fraction_leaf=0.0,max_depth=3,min_impurity_decrease=0.0,
                                           min_impurity_split=None,init=None, random_state=None,max_features=None, alpha=0.9,
                                           verbose=0, max_leaf_nodes=None,warm_start=False, presort='auto'
                                           , validation_fraction=0.1,n_iter_no_change=None, tol=0.0001)
'''

GBRModel = GradientBoostingRegressor(n_estimators=100,max_depth=2,learning_rate = 0.5 ,random_state=33)
GBRModel.fit(X_train, y_train)
#Calculating Details

print('GBRModel Train Score is : ' , GBRModel.score(X_train, y_train))
print('GBRModel Test Score is : ' , GBRModel.score(X_test, y_test))
#Calculating Prediction

y_pred = GBRModel.predict(X_test)
print('Predicted Value for GBRModel is : ' , y_pred[:10])
#Calculating Mean Absolute Error

from sklearn.metrics import mean_absolute_error 

MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue)
#Calculating Mean Squared Error

from sklearn.metrics import mean_squared_error 

MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)
#Calculating Median Squared Error

from sklearn.metrics import median_absolute_error

MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue )