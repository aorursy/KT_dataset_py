

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
mart_train = pd.read_csv('../input/Train.csv')

mart_test  = pd.read_csv('../input/Test.csv')
mart_train.head()
print(mart_train.shape, mart_test.shape)
# The shape of the training and test dataset shows that there are 8523 records in training and 5681 records in test dataset.
mart_train.info()
mart_test.info()
mart_train['Source']='Train'   # Creating a new column in train dataset and assigning value 'Train' in-order to classify the train dataset records after merging.

mart_test['Source'] = 'Test'   # Creating a new column in test dataset and assigning value 'Test' in-order to classify the test dataset records after merging.



full_data = pd.concat([mart_train, mart_test], ignore_index=True)



print(mart_train.shape, mart_test.shape, full_data.shape)
print(full_data.isna().sum())    # Getting the count of missing values in the full_data.
full_data.describe()
full_data.apply(lambda x : len(x.unique()))
# From the above result it's evident that the there are 1559 products in total and 16 distinct types of items. 

# Also, there are 10 outlets.

full_data.Item_Fat_Content.value_counts()
#Observe that "Low Fat," "LF", "low fat" are termed differently. Similarly, "Regular" and "Reg" as well. "LF" and "Low Fat" needs to be updated to "Low Fat"
full_data.Outlet_Location_Type.value_counts()
# Updating the null values of Item_Weight field with the mean

full_data['Item_Weight'].fillna(full_data['Item_Weight'].mean(), inplace= True)
# Check null values in Item_Weight

full_data['Item_Weight'].isna().sum()
# Updating the null values of Outlet_size with the mode

full_data['Outlet_Size'].fillna(full_data['Outlet_Size'].mode()[0], inplace=True)
# Check the null values in Outlet_Size

full_data['Outlet_Size'].isna().sum()
full_data.pivot_table(values='Item_Outlet_Sales', index = 'Outlet_Type')
full_data.loc[full_data['Item_Visibility']==0, 'Item_Visibility'] = full_data['Item_Visibility'].mean()
print((full_data['Item_Visibility']==0).sum())