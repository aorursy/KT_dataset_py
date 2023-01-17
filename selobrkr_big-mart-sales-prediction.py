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
import numpy as np #for linear algebra
import pandas as pd #for working with dataframe
import matplotlib.pyplot as plt #all for visualization
%matplotlib inline
import seaborn as sns
df_train = pd.read_csv('../input/big-mart-sales-train/datasets_9961_14084_Train_bigmart_sales.csv')
df_train.head()
df_train.info()
df_test = pd.read_csv('../input/big-mart-sales-test/datasets_9961_14084_Test_bigmart_sales.csv')
df_test.head()
df_test.info()
df_train['source'] = 'train' #we are putting extra columns to train and test data, we are going to use them in the end
df_test['source'] = 'test'
df_new = pd.concat([df_train, df_test])
df_new
df_new.info()
for i in df_new.describe().columns: #checking the distribution of numerical variables
    sns.distplot(df_new[i].dropna())
    plt.show()
for i in df_new.describe().columns:
    sns.boxplot(df_new[i].dropna())
    plt.show()
plt.figure(figsize=(15,6))
sns.countplot(df_new['Item_Type'], order=df_new['Item_Type'].value_counts().index)
plt.xticks(rotation=90)
print('Skewness: %f' %df_new['Item_Outlet_Sales'].skew())
print('Kurtosis: %f' %df_new['Item_Outlet_Sales'].kurt())
# Skewness is a measure of symmetry, or more precisely, the lack of symmetry. A distribution, or data set, 
# is symmetric if it looks the same to the left and right of the center point. Kurtosis is a measure of whether 
# the data are heavy-tailed or light-tailed relative to a normal distribution.
# For a symmetric normal distribution; skewness=0.3 and kurtosis=2.96
sns.distplot(df_new['Item_Outlet_Sales'])
g = sns.PairGrid(df_new)
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(plt.scatter)
df_new.corr()
df_new['Item_Fat_Content'].value_counts() #some of the variables are actually the same
df_new['Item_Fat_Content'].replace(to_replace = 'LF', value = 'Low Fat', inplace = True)
df_new['Item_Fat_Content'].replace(to_replace = 'low fat', value = 'Low Fat', inplace = True)
df_new['Item_Fat_Content'].replace(to_replace = 'reg', value = 'Regular', inplace = True)
df_new['Item_Fat_Content'].value_counts()
df_new['Item_Fat_Content']  = df_new['Item_Fat_Content'].map({'Low Fat':1, 'Regular':2}) #encoding
df_new['Item_Fat_Content'].value_counts()
df_new['Item_Identifier'].nunique() #there are 1559 unique variables
df_new['New_Item_Identifier'] = df_new['Item_Identifier'].apply(lambda x: x[0:2]) #we are trying to categorizing them
df_new['New_Item_Identifier'] = df_new['New_Item_Identifier'].map({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'})
df_new['New_Item_Identifier'].value_counts()
df_new['New_Item_Identifier'] = df_new['New_Item_Identifier'].map({'Food':1, 'Non-Consumable':2, 'Drinks':3}) #encoding
df_new['Item_Weight'] = df_new['Item_Weight'].fillna(df_new['Item_Weight'].mean()) #handling the missing values by filling
                                                                                   #with mean
df_new['Item_Type'].value_counts()
item_type_dummies = pd.get_dummies(data=df_new['Item_Type']) #making them dummy variables
df_new = pd.concat([df_new, item_type_dummies], axis=1)
outlet_identifier_dummies = pd.get_dummies(data=df_new['Outlet_Identifier']) #making them dummy variables
df_new = pd.concat([df_new, outlet_identifier_dummies], axis=1)
df_new['Outlet_Size'].isna().value_counts() #there are some missing values 
df_new['Outlet_Size'].value_counts()
sns.boxplot(data=df_new, x='Outlet_Size', y='Item_Outlet_Sales')
df_new['Outlet_Size'] = df_new['Outlet_Size'].fillna(value='Other') #named 'other' the missing values
df_new['Outlet_Location_Type_2'] = df_new['Outlet_Location_Type'].map({'Tier 1':1, 'Tier 2':2, 'Tier 3':3}) #encoding
df_new['Outlet_Type'].value_counts()
df_new['Outlet_Type_2'] = df_new['Outlet_Type'].map({'Supermarket Type1':1, 'Supermarket Type2':2, 'Supermarket Type3':3,
                                                        'Grocery Store': 4}) #encoding
df_new.columns.to_list
df_new.select_dtypes(include=[np.object]) #checking the object type variables
df_new = df_new.drop(['Item_Identifier', 'Item_Type', 'Outlet_Identifier', 'Outlet_Location_Type', 'Outlet_Type'], 
                         axis=1) #dropping the object type variables
df_new['Outlet_Size'].value_counts()
df_new['Outlet_Size_2'] = df_new['Outlet_Size'].map({'Small':1, 'Medium':2, 'High':3, 'Other':4}) #encoding
df_new = df_new.drop(['Outlet_Size'], axis=1) #dropping the object type variables after encoding
df_new.head()
df_new['Outlet_Establishment_Year'].value_counts()
establishment_year = pd.get_dummies(data=df_new['Outlet_Establishment_Year']) #making dummy variables
establishment_year
df_new = pd.concat([df_new,establishment_year], axis=1)
df_new = df_new.drop('Outlet_Establishment_Year', axis=1)
df_new.columns
df_new.isna().any() #last check before modelling
train = df_new.loc[df_new['source'] == 'train'] #splitting train and test data using 'source' column we created in the
                                                #beginning
test = df_new.loc[df_new['source'] == 'test']
train = train.drop(['source'], axis=1)
test = test.drop(['Item_Outlet_Sales', 'source'], axis=1)
X_train = train.drop(['Item_Outlet_Sales'], axis=1)
y_train = train['Item_Outlet_Sales']
X_test = test.copy()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
pred = lm.predict(X_test)
pred
pred.shape
y_train.shape
plt.figure(figsize=(12,6))
plt.plot(pred)
plt.plot(y_train)
plt.show()
accuracy = lm.score(X_train, y_train)
round(accuracy, ndigits=2)
