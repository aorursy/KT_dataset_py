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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as mode

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb



# Setting a figure size of all the graphs at first

plt.rcParams['figure.figsize'] = 12,8

plt.style.use('ggplot')



# Disabling warnings

import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('/kaggle/input/bigmart-sales-data/Train.csv')

df_test = pd.read_csv('/kaggle/input/bigmart-sales-data/Test.csv')
print('Train set shape : ', df_train.shape)

print('Test set shape : ', df_test.shape)
df_train.head(4)
df_train.describe()
df_train.info()
t = df_train.groupby(['Outlet_Location_Type'])['Item_Outlet_Sales', 'Item_MRP'].mean()

t.style.background_gradient(cmap = 'BuPu')
ob = df_train.groupby('Outlet_Location_Type')



for name,group in ob:

    print(name, 'contains', group.shape[0], 'rows')
ob.get_group('Tier 2').head(10)
sns.distplot(df_train.Item_Outlet_Sales, bins=50, color = 'red')

plt.xlabel('Item Outlet Sales')

plt.ylabel('No. of scales')

plt.title('Item Outlet Sales distribution')
num_feature = df_train.select_dtypes(include=[np.number])

num_feature.dtypes
num_feature.corr()
plt.style.use('bmh')

print(num_feature.corr()['Item_Outlet_Sales'].sort_values(ascending=False))

sns.heatmap(num_feature.corr(), annot=True, cmap = 'magma');
plt.plot(df_train['Item_MRP'], df_train['Item_Outlet_Sales'], '_', color = '#E2A50F')

plt.xlabel('Item MRP')

plt.ylabel('Sales')

plt.title('Impact of MRP of an Item on Sales');
# Checking all categorical variables

cat_features = df_train.select_dtypes(include=[np.object])

cat_features.dtypes
print(df_train.Item_Identifier.value_counts().head(10))

print()

print('No. of Unique itmes : ', df_train.Item_Identifier.nunique())
df_train.Item_Fat_Content.value_counts()
df_train['Item_Fat_Content'] = df_train['Item_Fat_Content'].replace('LF','Low Fat')

df_train['Item_Fat_Content'] = df_train['Item_Fat_Content'].replace('low fat', 'Low Fat')

df_train['Item_Fat_Content'] = df_train['Item_Fat_Content'].replace('reg', 'Regular')
plt.style.use('bmh')

color = '#7FD59E', '#D57FB3'

plt.pie(df_train.Item_Fat_Content.value_counts(), data = df_train, labels = ('Low Fat', 'Regular Fat'), 

        colors = color, autopct='%1.1f%%', startangle = 70, shadow = True, explode = (0.08, 0), textprops={'fontsize': 16})

plt.title('Fat Content in items');
# Lets see if fat content has any impact on the sales

Fat_cont_pivot = df_train.pivot_table(index = 'Item_Fat_Content', values = 'Item_Outlet_Sales', aggfunc = np.mean)

print(Fat_cont_pivot)



Fat_cont_pivot.plot(kind = 'bar', color = '#91992C')

plt.xlabel('Item_Fat_Content')

plt.xticks(rotation = 0)

plt.ylabel('Item_Outlet_sales')

plt.title('Impact of Fat content on Sales');
print(df_train['Item_Type'].value_counts())

sns.countplot(y=df_train['Item_Type'], palette = 'tab20b');
Item_type_pivot = df_train.pivot_table(index = 'Item_Type', values = 'Item_Outlet_Sales', aggfunc = np.mean)

print(Item_type_pivot)



Item_type_pivot.plot(kind = 'bar', color = '#91992C')

plt.xlabel('Item Type')

plt.xticks(rotation = 90)

plt.ylabel('Item_Outlet_sales')

plt.title('Impact of Item types on Sales');
print(df_train['Outlet_Size'].value_counts())

sns.countplot(x=df_train['Outlet_Size'], palette = 'cubehelix');
out_size_pivot = df_train.pivot_table(index = 'Outlet_Size', values = 'Item_Outlet_Sales', aggfunc = np.mean)

print(out_size_pivot)



out_size_pivot.plot(kind = 'bar', color = '#91992C')

plt.xlabel('Outlet Size')

plt.xticks(rotation = 0)

plt.ylabel('Item_Outlet_sales')

plt.title('Impact of Outlet Size on Sales');
print(df_train['Outlet_Location_Type'].value_counts())

sns.countplot(x=df_train['Outlet_Location_Type'], palette = 'viridis');
out_loc_pivot = df_train.pivot_table(index = 'Outlet_Location_Type', values = 'Item_Outlet_Sales', aggfunc = np.mean)

print(out_loc_pivot)



out_loc_pivot.plot(kind = 'bar', color = '#91992C')

plt.xlabel('Outlet Location Type')

plt.xticks(rotation = 0)

plt.ylabel('Item_Outlet_sales')

plt.title('Impact of Location Type on Sales');
print(df_train['Outlet_Type'].value_counts())

sns.countplot(x=df_train['Outlet_Type'], palette = 'rocket_r');
out_type_pivot = df_train.pivot_table(index = 'Outlet_Type', values = 'Item_Outlet_Sales', aggfunc = np.mean)

print(out_type_pivot)



out_type_pivot.plot(kind = 'bar', color = '#91992C')

plt.xlabel('Outlet Type')

plt.xticks(rotation = 0)

plt.ylabel('Item_Outlet_sales')

plt.title('Impact of Outlet Type on Sales');
df_train.isnull().sum()
df_test.isnull().sum()
# We can simply impute the missing values of Item weight by its mean.

df_train['Item_Weight'].fillna(df_train['Item_Weight'].mean(), inplace=True)

df_test['Item_Weight'].fillna(df_test['Item_Weight'].mean(), inplace=True)
# Determing the mode for each -Train Data

out_size_mode_train = df_train.pivot_table(values = 'Outlet_Size', columns = 'Outlet_Type', aggfunc = lambda x: x.mode())
# Determing the mode for each -Test Data

out_size_mode_test = df_test.pivot_table(values = 'Outlet_Size', columns = 'Outlet_Type', aggfunc = lambda x: x.mode())
def nan_out_size_train(cols):

    size = cols[0]

    Type = cols[1]

    if pd.isnull(size):

        return out_size_mode_train.loc['Outlet_Size'] [out_size_mode_train.columns == Type][0]

    else:

        return size

    

df_train['Outlet_Size'] = df_train[['Outlet_Size', 'Outlet_Type']].apply(nan_out_size_train, axis = 1)
def nan_out_size_test(cols):

    size = cols[0]

    Type = cols[1]

    if pd.isnull(size):

        return out_size_mode_test.loc['Outlet_Size'] [out_size_mode_test.columns == Type][0]

    else:

        return size

    

df_test['Outlet_Size'] = df_test[['Outlet_Size', 'Outlet_Type']].apply(nan_out_size_test, axis = 1)
agg_visible = df_train.pivot_table(values='Item_Visibility', index='Item_Identifier')

agg_visible
def visible_mean(cols):

    visible = cols[0]

    items = cols[1]

    if visible == 0:

        return agg_visible['Item_Visibility'][agg_visible.index ==items]

    else:

        return visible

    

df_train[['Item_Visibility', 'Item_Identifier']].apply(visible_mean, axis = 1).astype(float)
# For Training data

funct_tr = lambda x: x['Item_Visibility']/agg_visible['Item_Visibility'][agg_visible.index == x['Item_Identifier']][0]

df_train['Item_Vis_Mean'] = df_train.apply(funct_tr,axis=1).astype(float)

df_train['Item_Vis_Mean'].describe()
# For Test Data

funct_ts = lambda x: x['Item_Visibility']/agg_visible['Item_Visibility'][agg_visible.index == x['Item_Identifier']][0]

df_test['Item_Vis_Mean'] = df_test.apply(funct_ts,axis=1).astype(float)

df_test['Item_Vis_Mean'].describe()
# Let's Consider the outlet establishment years

print(df_train['Outlet_Establishment_Year'].unique())

print()

print(df_test['Outlet_Establishment_Year'].unique())
# The data we have is from year 2013, so we would consider 2013 for calculating that how old is the outlet

df_train['Outlet_years'] = 2013 - df_train['Outlet_Establishment_Year']

df_test['Outlet_years'] = 2013 - df_test['Outlet_Establishment_Year']
df_train['Outlet_years'].describe()
df_test['Outlet_years'].describe()
# For Train data

df_train['Combined_Item_Type'] = df_train['Item_Identifier'].apply(lambda x: x[0:2])



df_train['Combined_Item_Type'] = df_train['Combined_Item_Type'].map({'FD':'Food',

                                                             'NC':'Non-Consumable',

                                                             'DR':'Drinks'})

df_train['Combined_Item_Type'].value_counts()
# For Test data

df_test['Combined_Item_Type'] = df_test['Item_Identifier'].apply(lambda x: x[0:2])



df_test['Combined_Item_Type'] = df_test['Combined_Item_Type'].map({'FD':'Food',

                                                             'NC':'Non-Consumable',

                                                             'DR':'Drinks'})

df_test['Combined_Item_Type'].value_counts()
df_train['Item_Fat_Content'] = df_train['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
df_train.loc[df_train['Combined_Item_Type'] == 'Non-Consumable', 'Item_Fat_Content'] = 'In-edible'

df_train['Item_Fat_Content'].value_counts()
df_test['Item_Fat_Content'] = df_test['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
df_test.loc[df_test['Combined_Item_Type'] == 'Non-Consumable', 'Item_Fat_Content'] = 'In-edible'

df_test['Item_Fat_Content'].value_counts()
label_encode = LabelEncoder()

#New variable for outlet

df_train['Outlet'] = label_encode.fit_transform(df_train['Outlet_Identifier'])

var_mod_train = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Combined_Item_Type','Outlet_Type','Outlet']

for i in var_mod_train:

    df_train[i] = label_encode.fit_transform(df_train[i])
label_encode = LabelEncoder()

#New variable for outlet

df_test['Outlet'] = label_encode.fit_transform(df_test['Outlet_Identifier'])

var_mod_test = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Combined_Item_Type','Outlet_Type','Outlet']

for i in var_mod_test:

    df_test[i] = label_encode.fit_transform(df_test[i])
#Dummy Variables:

df_train = pd.get_dummies(df_train, columns =['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',

                                  'Combined_Item_Type','Outlet'])





df_test = pd.get_dummies(df_test, columns =['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',

                                  'Combined_Item_Type','Outlet'])

df_train.drop(df_train[['Item_Type', 'Item_Identifier', 'Outlet_Identifier']], axis=1, inplace=True)

df_test.drop(df_test[['Item_Type', 'Item_Identifier', 'Outlet_Identifier']], axis=1, inplace=True)
print(df_train.shape)

print(df_test.shape)
x = df_train.drop(['Item_Outlet_Sales'], axis=1).values

y = df_train.Item_Outlet_Sales.values
train_x, test_x, train_y, test_y = train_test_split(x,y, test_size = 0.2, random_state = 42)

train_x.shape, test_x.shape, train_y.shape, test_y.shape
feature_scale = StandardScaler()

train_x = feature_scale.fit_transform(train_x)

test_x = feature_scale.transform(test_x)
#Linear Regression

lin_reg = LinearRegression(normalize=True)

lin_reg.fit(train_x,train_y)

print('Root Mean Squared Error : ', np.sqrt(metrics.mean_squared_error(test_y, lin_reg.predict(test_x))))
#Random Forest

rf = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_leaf=100, random_state=42)

rf.fit(train_x, train_y)



print('Root Mean Squared Error : ', np.sqrt(metrics.mean_squared_error(test_y, rf.predict(test_x))))
#Decision Tree

DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)

DT.fit(train_x, train_y)



print('Root Mean Squared Error : ', np.sqrt(metrics.mean_squared_error(test_y, DT.predict(test_x))))
#Xtreme Gradient Boosting

boost = xgb.XGBRegressor(learning_rate = 0.01, n_estimators=1000, max_depth = 4, random_state = 42)

boost.fit(train_x, train_y)



print('Root Mean Squared Error : ', np.sqrt(metrics.mean_squared_error(test_y, boost.predict(test_x))))
test_pred_rf = rf.predict(df_test)

test_pred_rf