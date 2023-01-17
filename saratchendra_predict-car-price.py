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
# Importing the required Libraries

import pandas as pd             # Data Processing (Ex: read, merge)  

import numpy as np              # For mathemetical calculations

import seaborn as sns           # For Data visulization

import matplotlib.pyplot as plt # For ploting the graphs
## Load the data

df=pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')
df.head()  # To find top 5 columns from Data
df.info()  # To find the information
df.shape   # To check number of rows and columns
df.nunique()     # To see number of unique values(Features) in every individual columns
df.isna().sum()   # To check the missing values
df.describe().drop('count').T  # returns some common statistical details of the data
df1=df.copy()          # Copy the data into df1
df1['Car_Name'].nunique()  # There are 98 different Cars
# Seperate the categorical and numerical variable 

categorical_columns=df1.columns[df.dtypes=='object']

numerical_columns=df1.columns[df.dtypes!='object']

print(numerical_columns)

print(categorical_columns)
# Let us run through the categorical variable

print(df1['Seller_Type'].value_counts(), '\n') 

print(df1['Transmission'].value_counts(), '\n')

print(df1['Owner'].value_counts())
## It is observed that in 'Owner' column there are three categories 0,1,3. Lets replace 3 with 1 

df1['Owner']=df1['Owner'].replace(3, 1) 

print(df1['Owner'].value_counts())
# Let us find the Price difference between Present price and Selling price

df1['Price_diff']=df['Present_Price']-df['Selling_Price']
df1.head()
sns.pairplot(df1)
# Visualization is the best way to understand and analyse the data

plt.figure(figsize=(13,7))

df1['Year'].value_counts().plot.bar()
plt.figure(figsize=(15,8))

sns.barplot(x='Year',y='Selling_Price', data=df1)
# From the above graph it is observed that Car Selling was low in 2007 and high in 2018
# The simple way is to use for loop to go through all the columns
# Plotting the histogram for numerical column

plt.figure(figsize=(20,20))

for i in range(len(numerical_columns)):

    plt.subplot(4,2,i+1)

    plt.hist(df1[numerical_columns[i]], bins=30)
# Plotting the distplot for numerical column

plt.figure(figsize=(20,20))

for i in range(len(numerical_columns)):

    plt.subplot(4,2,i+1)

    sns.distplot(df1[numerical_columns[i]], kde_kws = {'bw' : 1})
plt.figure(figsize=(13,7))

sns.barplot(x='Fuel_Type', y='Selling_Price', data=df1, palette = "gist_rainbow_r")
# From the above graph it is observed that Diesel cars were more expensive than Petrol and CNG
plt.figure(figsize=(13,7))

sns.barplot(x='Fuel_Type', y='Selling_Price',hue='Transmission', data=df1, palette = "BuGn")
# From the above graph it is observed that CNG does not have any Transmission i.e., Manual and Automatic

## Fuel_Type Diesel has more Automatic transmission than Petrol
plt.figure(figsize=(13,7))

sns.barplot(x='Fuel_Type', y='Selling_Price',hue='Owner', data=df1, palette = "nipy_spectral")
df1.corr()
# There is a correlation between Selling price and Present price. 
plt.figure(figsize=(15,7))

sns.heatmap(df1.corr(), annot=True, cmap='Blues')
df1=df1.drop(['Year', 'Car_Name'], 1)
df1.head()
df1=pd.get_dummies(df1, drop_first=True)
df1.head()
df1.corr()
# There is a strong correlation between Selling price and Fuel_Type_Diesel.

# There is a correlation between Price diff and Selling Price.

# There is a correlation between Price diff and Selling Price.

plt.figure(figsize=(13,7))

sns.heatmap(df1.corr(), annot=True)
X=df1.drop('Selling_Price',1)

y=df1['Selling_Price']
### Feature Importance



from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt

model = ExtraTreesRegressor()

model.fit(X,y)
print(model.feature_importances_)
#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(5).plot(kind='barh')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LinearRegression

lr= LinearRegression()
lr.fit(X_train, y_train)
pred=lr.predict(X_test)
plt.scatter(y_test,pred)
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
regressor.fit(X_train, y_train)
pred=regressor.predict(X_test)
sns.distplot(y_test-pred)

plt.scatter(y_test,pred)
from sklearn.tree import DecisionTreeRegressor

dt= DecisionTreeRegressor()
dt.fit(X_train, y_train)
pred=regressor.predict(X_test)
plt.scatter(y_test, pred)
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, pred))

print('MSE:', metrics.mean_squared_error(y_test, pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))