import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statistics as st

import seaborn as sns

%matplotlib inline

import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)
train_data = pd.read_csv('../input/big-mart-sales-dataset/Train_UWu5bXk.csv')

train_data
train_data.describe(include='all')
train_data.info()
test_data = pd.read_csv('../input/big-mart-sales-dataset/Test_u94Q5KV.csv')

test_data
test_data.info()
# save the target attribute

label = train_data['Item_Outlet_Sales']

# combining both data sets

train_data.drop('Item_Outlet_Sales', axis=1, inplace=True)
combined = train_data.append(test_data)
combined.reset_index(inplace=True)
combined.drop('index',axis=1,inplace=True)
# whole combined data set 

combined
combined.info()
# changning the type of attributes

combined.Item_Fat_Content.value_counts()

combined.Item_Fat_Content = combined.Item_Fat_Content.astype('category')

combined.Item_Type = combined.Item_Type.astype('category')

combined.Outlet_Identifier = combined.Outlet_Identifier.astype('category')

combined.Outlet_Size = combined.Outlet_Size.astype('category')

combined.Outlet_Location_Type = combined.Outlet_Location_Type.astype('category')

combined.Outlet_Type = combined.Outlet_Type.astype('category')

combined.Item_Identifier = combined.Item_Identifier.astype('category')
# now separate the qualitative and quantitative data

numeric_data = combined.select_dtypes(exclude=['category'])

cat_data = combined.select_dtypes(include=['category'])
# first deal with numeric data

numeric_data.info()
# cleaning and scaling the numeric values

# handle missing values

# applying feature scaling to numeric data

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



#scpy.stat skew function, on dependent variable

# p.log1p() # dependent variable

# box coks1p  # independent
num_pip = Pipeline([

    ('imputer', Imputer(strategy='median')),

    ('std_scaler', StandardScaler())

])

numeric_clean_data = num_pip.fit(numeric_data).transform(numeric_data)
numeric_clean_data = pd.DataFrame(numeric_clean_data,columns=numeric_data.columns)

numeric_clean_data
# cleaning and handling categorical data

cat_data.Item_Fat_Content.value_counts()
cat_data.Item_Fat_Content = cat_data.Item_Fat_Content.replace('LF','Low Fat')

cat_data.Item_Fat_Content = cat_data.Item_Fat_Content.replace('reg','Regular')

cat_data.Item_Fat_Content = cat_data.Item_Fat_Content.replace('low fat','Low Fat')
cat_data.Item_Fat_Content.value_counts()
#outlet size

cat_data.isnull().sum()
cat_data=cat_data.fillna(

    {'Outlet_Size':st.mode(cat_data.Outlet_Size)}

)

cat_data.Outlet_Size.isnull().sum()
cat_data.isnull().sum() # categorical columns have been cleaned
# now one hot encoding of categorical variables

cat_coding_data = pd.get_dummies(cat_data,columns=['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size'

                                                  ,'Outlet_Location_Type','Outlet_Type'])
cat_coding_data
# now combine the both numerical and categorical data

process_data = pd.concat([numeric_clean_data,cat_coding_data,label],axis=1)
train_data = process_data[:8523]

test_data = process_data[8523:]

# first remove outliers an than normalize
train_data

test_data
# # now both numerical and categorical data have been cleaned

# # its time to check outliers in the target(dependent) variable

# remove outlier in train data

train_data.Item_Outlet_Sales.plot.box()

train_data.shape
def reject_outliers(data):

    u = np.median(data)

    s = np.std(data)

    filtered = [e for e in data if (u - 4.5 * s < e < u + 4.5 * s)]

    return filtered



# Item_Outlet_Size is my dependent variable

label_data1 = reject_outliers(train_data.Item_Outlet_Sales)
label_data2 = np.array(label_data1)
label_data4 = pd.DataFrame(label_data2)
# label_data4
# outliers have been removed from data dependent variable

label_data4.plot.box()

label_data4.shape
# # now we get the data by removing outliers



arr_2d = label_data4.values

arr_1d = arr_2d.ravel()

arr_lst = list(arr_1d)

train_data_cleaned = process_data.loc[process_data.Item_Outlet_Sales.isin(arr_lst)]

train_data_cleaned.shape
train_data_cleaned.reset_index(inplace=True)
train_data_cleaned.drop('index', axis=1, inplace=True)
train_label = train_data_cleaned.Item_Outlet_Sales

train_features = train_data_cleaned.drop(['Item_Outlet_Sales','Item_Identifier'],axis=1)
train_data_cleaned.to_csv("train_big_mart_data.csv")
test_data.drop('Item_Outlet_Sales',axis = 1, inplace=True)
test_data.reset_index(inplace=True)
test_data.drop('index',inplace=True,axis=1)
# test_data.drop('level_0',inplace=True,axis=1)
test_data.to_csv('test_big_mart_data.csv')