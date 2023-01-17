# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_rows', None)
data = pd.read_csv('../input/train.csv')
data.head()
plt.figure(figsize=(10,8))

sns.heatmap(data.isnull())
data.isnull().sum()
data.describe(include='all')
data['Item_Fat_Content'].unique()
sns.countplot(x=data['Item_Fat_Content'], data=data)
def replace_item(item):

    if (item == 'Low Fat'):

        return 'LF'

    elif (item == 'Regular'):

        return 'RE'

    elif item == 'low fat' :

        return 'LF'

    elif item == 'reg':

        return 'RE'

    elif item == 'LF':

        return 'LF'

    else :

        return 'OT'
#data['Item_Fat_Content'] = data['Item_Fat_Content'].map({'Low Fat':'LF','Regular':'RE','low fat':'LF','reg':'RE', np.NaN:0})

data['Item_Fat_Content'] = data['Item_Fat_Content'].apply(replace_item)
data['Item_Fat_Content'].unique()
data.head()
sns.boxplot(x=data['Item_Fat_Content'], y=data['Item_Weight'], data=data)
data['Item_Type'].unique()
plt.figure(figsize=(25,5))

sns.boxplot(x=data['Item_Type'], y=data['Item_Weight'], data=data)
#data[['Item_Fat_Content','Item_Weight','Item_Type']].apply(null_imputer)

data2 = data.copy()
data2.isnull().sum()
def null_imputer(item):

    item_weight = item[0]

    item_type = item[1]

    item_fat = item[2]

    if pd.isnull(item_weight) :

        return 12.5

    else:

        return item_weight
#data[['Item_Fat_Content','Item_Weight','Item_Type']].apply(null_imputer)

data2['Item_Weight'] = data2[['Item_Weight','Item_Type','Item_Fat_Content']].apply(null_imputer, axis=1)
data2.isnull().sum()
temp_df = data2[data2['Outlet_Size'].isnull()]
temp_df.head()
print("Missing Values from Tier: ", temp_df['Outlet_Location_Type'].unique())

print("Missing Values from Outlet Type: ", temp_df['Outlet_Type'].unique())
temp2 = data2[data2['Outlet_Location_Type']=='Tier 2']
print("We are missing the Outlet Size values for Market Type: ",temp2[temp2['Outlet_Size'].isnull()]['Outlet_Type'].unique())
temp2[temp2['Outlet_Type']=='Supermarket Type1'].head()
temp3 = data2[data2['Outlet_Location_Type']=='Tier 3']
#Checking for what outlet types and Supermarket types we are actually missing the Outlet size values:

print("We are missing the Outlet Size values for Market Type: ",temp3[temp3['Outlet_Size'].isnull()]['Outlet_Type'].unique())
data2.isnull().sum()
def size_imputer(item):

    size = item[0]

    tier = item[1]

    out_type = item[2]

    sales = item[3]

    

    if pd.isnull(size) :

        if tier == 'Tier 3' :

            if out_type == 'Grocery Store':

                return 'Low'

            elif out_type == 'Supermarket Type1':

                return 'High'

            else:

                return 'Medium'

        elif tier == 'Tier 2':

            if out_type == 'Supermarket Type1':

                if sales < 8479.6288 :

                    return 'Low'

                else:

                    return 'Medium'

        else:

            return size

    else:

        return size
data3 = data2.copy()
data3['Outlet_Size'] = data3[['Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Outlet_Sales']].apply(size_imputer, axis=1)
data3.isnull().sum()
data4 = data3.copy()
fat_df = pd.get_dummies(data4['Item_Fat_Content'], drop_first=True, prefix='Item_Fat_Content')
item_type_df = pd.get_dummies(data4['Item_Type'], drop_first=True, prefix='Item_Type')
outlet_size_df = pd.get_dummies(data4['Outlet_Size'], drop_first=True, prefix='Outlet_Size')
outlet_type_df = pd.get_dummies(data4['Outlet_Type'], drop_first=True, prefix='Outlet_Type')
tier_df = pd.get_dummies(data4['Outlet_Location_Type'], drop_first=True)
final_df = pd.concat([data4[['Item_Weight','Item_Visibility','Item_MRP','Item_Outlet_Sales']],fat_df,item_type_df,outlet_size_df,

                     outlet_type_df,tier_df], axis=1)
X = final_df.drop(['Item_Outlet_Sales'], axis=1)

y = final_df['Item_Outlet_Sales']
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X,y)
rfr.score(X,y)
## Using the training data itself by splitting it and testing it
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)
rfr2 = RandomForestRegressor()
rfr2.fit(X_train, y_train)
rfr2.score(X_train, y_train)
predictions = rfr2.predict(X_test)
from sklearn import metrics
print(metrics.mean_absolute_error(y_test, predictions))
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#This is a really good score
original_target_df = y_test.to_frame()
original_target_df.reset_index(inplace=True)
original_target_df.drop('index',axis=1, inplace=True)
pred_df = pd.DataFrame(data=predictions, columns=['Predicted Sales'])
compare_df = pd.concat([original_target_df, pred_df], axis=1, ignore_index=True)
compare_df.columns = ['Actual Values','Predicted Values']
compare_df.head()