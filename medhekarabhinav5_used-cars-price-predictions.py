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
train_data = pd.read_csv("/kaggle/input/used-cars-price-prediction/train-data.csv")

test_data = pd.read_csv("/kaggle/input/used-cars-price-prediction/test-data.csv")
train_data.rename(columns = {'Unnamed: 0' : 'Id'}, inplace = True)

test_data.rename(columns = {'Unnamed: 0' : 'Id'}, inplace = True)

train_data.set_index('Id', inplace = True)

test_data.set_index('Id', inplace = True)
train_data.head()
test_data.head()
train_data.isnull().sum()
train_data.describe()
test_data.isnull().sum()
test_data.describe()
y = train_data['Price']

X = train_data.drop(['Price'], axis = 1)
X
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2)
X_train.count()
X_valid.count()
y_train.count()
y_valid.count()
X_train.columns
X_train.dtypes
#Lets first check cardinality of columns in X_train data



X_train.nunique()
X_train['Location'].unique()
len(X_train['Location'].unique())
import matplotlib.pyplot as plt

import seaborn as sns

def pie_countplot(dataset, column, title, explosion_rate):

    column_length = len(dataset[column].unique())

    

                            #n_rows, n_cols, figure_size

    f, plt_size = plt.subplots(1, 2, figsize = (25,10))

    dataset[column].value_counts().plot.pie(explode = [explosion_rate for i in range(column_length)], autopct = '%1.1f%%', ax = plt_size[0], shadow = False)

    plt_size[0].set_title(title)

    plt_size[0].set_ylabel('')



    sns.countplot(column, data = dataset, ax = plt_size[1])

    plt_size[1].set_title(title)

    plt.show()
pie_countplot(X_train,'Location', 'Locations', 0.01)
pie_countplot(X_train,'Year', 'Years', 0.01)
pie_countplot(X_train, 'Fuel_Type', 'Fuel Types', 0.01)
pie_countplot(X_train, 'Transmission', 'Car Transmissions', 0.01)
pie_countplot(X_train, 'Owner_Type', 'Number of Owners', 0.01)
X_train.nunique()
X_train.dtypes
def bar_countplot(dataset, column1, column2, title):

                            #n_rows, n_cols, figure_size

    f, plt_size = plt.subplots(1, 2, figsize = (25,10))

    dataset[[column1, column2]].groupby([column2]).mean().plot.bar(ax = plt_size[0])

    plt_size[0].set_title(title)

    sns.countplot(column1, hue = column2, data = dataset, ax = plt_size[1])

    plt_size[1].set_title(title)

    plt.show()
#bar_countplot(X_train, 'Engine', 'Mileage', 'Engine vs. Mileage')
# Number of missing values per Column

X_train.isnull().sum()
X_test = test_data.copy()

X_test.isnull().sum()
# a sample data check 

X_train.head(1)
X_test.head(1)
X_train['Seats'].unique()
X_valid['Fuel_Type'].unique()
# Lets start by Changing Location to Numerical

# first start with label encoding. There are some columns which have low cardinality and less number of missing values

from sklearn.preprocessing import LabelEncoder



my_Label_Encoder = LabelEncoder()

def labelEncoding(column):

    X_train[column] = my_Label_Encoder.fit_transform(X_train[column])

    my_Label_Encoder_mapping = dict(zip(my_Label_Encoder.classes_, my_Label_Encoder.transform(my_Label_Encoder.classes_)))

    print(my_Label_Encoder_mapping)

    X_valid[column] = my_Label_Encoder.transform(X_valid[column])

    X_test[column] = my_Label_Encoder.transform(X_test[column])
labelEncoding('Fuel_Type')
labelEncoding('Location')
labelEncoding('Transmission')
labelEncoding('Owner_Type')
X_train.head()
X_valid.head()
def column_extract_num(dataset, old_column, new_column):

    dataset[new_column] = dataset[old_column].str.extract('(\d*\.\d+|\d+)')

    dataset.drop([old_column], axis = 1, inplace = True)
column_extract_num(X_train, 'Mileage', 'Fuel_Economy')

column_extract_num(X_train, 'Engine', 'Engine_CC')

column_extract_num(X_train, 'Power', 'Power_BHP')

column_extract_num(X_train, 'New_Price', 'Original_Price')
column_extract_num(X_valid, 'Mileage', 'Fuel_Economy')

column_extract_num(X_valid, 'Engine', 'Engine_CC')

column_extract_num(X_valid, 'Power', 'Power_BHP')

column_extract_num(X_valid, 'New_Price', 'Original_Price')
column_extract_num(X_test, 'Mileage', 'Fuel_Economy')

column_extract_num(X_test, 'Engine', 'Engine_CC')

column_extract_num(X_test, 'Power', 'Power_BHP')

column_extract_num(X_test, 'New_Price', 'Original_Price')
X_valid.head(1)
X_train.drop(columns = ['Name'], inplace = True)

X_valid.drop(columns = ['Name'], inplace = True)

X_test.drop(columns = ['Name'], inplace = True)
X_train.head(1)
X_train.isnull().sum()
X_train.columns
X_train['Original_Price'].isnull().sum()
from sklearn.impute import SimpleImputer



my_Simple_Imputer = SimpleImputer()

X_train_imputed = pd.DataFrame(my_Simple_Imputer.fit_transform(X_train))

X_valid_imputed = pd.DataFrame(my_Simple_Imputer.transform(X_valid))

X_test_imputed = pd.DataFrame(my_Simple_Imputer.transform(X_test))
X_train_imputed.columns = X_train.columns

X_valid_imputed.columns = X_valid.columns

X_test_imputed.columns = X_test.columns
X_train_imputed.isnull().sum()
test_data_chart = X_train_imputed

test_data_chart.columns = X_train_imputed.columns
y_train.isnull().sum()
test_data_chart
test_data_chart.isnull().sum()
y_train.min()
y_train.max()
sns.heatmap(test_data_chart.corr(), annot=True, cmap='Blues',linewidth=0.2)

fig = plt.gcf()

fig.set_size_inches(10,8)

plt.show()
def sns_plot(column1, column2):

    sns.lmplot(x=column1, y=column2, data=test_data_chart)
#year - Fuel_Economy ( 0.32 )

sns_plot('Year','Fuel_Economy')
#transmission - Fuel Economy ( 0.34 )

sns_plot('Fuel_Economy', 'Transmission')
#Seats - Engine_CC ( 0.37 )

sns_plot('Engine_CC','Seats')
#Location - Fuel Type ( 0.12 )

sns_plot('Location', 'Fuel_Type')
#Fuel_type - Transmission ( 0.12 )

sns_plot('Fuel_Type', 'Transmission')
#Engine_CC - Power_BHP ( 0.86 )

sns_plot('Engine_CC', 'Power_BHP')
#Engine_CC - Original_Price ( 0.21 )

sns_plot('Engine_CC', 'Original_Price')
#Power_BHP - Original_Price ( 0.27 )

sns_plot('Power_BHP', 'Original_Price')
#Year - Owner_Type ( -0.38 )

sns_plot('Year', 'Owner_Type')
#Kilometer_Driven - Year ( -0.15 )

sns_plot('Kilometers_Driven', 'Year')
#Fuel_Type - Fuel_Economy

sns_plot('Fuel_Economy', 'Fuel_Type')
#Fuel_Type - Seats

sns_plot('Fuel_Type', 'Seats')
X_train_imputed.columns
#Fuel_Type - Engine_CC ( -0.39 )

sns_plot('Fuel_Type', 'Engine_CC')
#Fuel_Type - Power_BHP ( -0.25 )

sns_plot('Fuel_Type', 'Power_BHP')
#Fuel_Economy - Engine_CC

sns_plot('Fuel_Economy','Engine_CC')
#Fuel_Economy - Power_BHP ( -0.51 )

sns_plot('Fuel_Economy', 'Power_BHP')
X_train_imputed.columns
X_train_imputed.isnull().sum()
y_train.isnull().sum()
X_valid_imputed.isnull().sum()
X_valid_imputed.columns
y_valid.isnull().sum()
X_test_imputed.isnull().sum()
# First we'll try to implement Desicion tree



from sklearn.tree import DecisionTreeRegressor

DT_Model = DecisionTreeRegressor(random_state = 1)



# We will train X_train_imputed and y_train.

DT_Model.fit(X_train_imputed, y_train)

DT_Model_Prds = DT_Model.predict(X_valid_imputed)
# checking Mean Absolute Error

from sklearn.metrics import mean_absolute_error



def get_mae(preds, vals):

    Mae = mean_absolute_error(DT_Model_Prds, y_valid)

    print(Mae)
get_mae(DT_Model_Prds, y_valid)
from sklearn.ensemble import RandomForestRegressor

RF_Model = RandomForestRegressor(n_estimators = 100 , random_state = 1)



RF_Model.fit(X_train_imputed, y_train)

RF_Model_Preds = RF_Model.predict(X_valid_imputed)



get_mae(RF_Model_Preds, y_valid)
from xgboost import XGBRegressor

XGB_Model = XGBRegressor(n_estimators = 500 , learning_rate = 0.2, random_state = 1)



XGB_Model.fit(X_train_imputed, y_train)

XGB_Model_Preds = XGB_Model.predict(X_valid_imputed)



get_mae(XGB_Model_Preds, y_valid)
X_final = pd.concat([X_train_imputed, X_valid_imputed,], axis = 0)
y_final = pd.concat([y_train, y_valid], axis = 0)
X_final.count()
y_final.count()
XGB_Model.fit(X_final, y_final)

XGB_final_preds = XGB_Model.predict(X_test_imputed)
XGB_final_preds.round(2)

test_data.reset_index(inplace = True)

XGB_final = pd.DataFrame({'Id' : test_data.Id, 'Pred Price' : XGB_final_preds})

XGB_final.set_index('Id', inplace = True)
XGB_final