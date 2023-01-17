# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
start=time.time()
train_df=pd.read_csv("../input/Train.csv")
test_df=pd.read_csv("../input/Test.csv")
end=time.time()
print("time taken by this cell: {}".format(end-start))
print("no. of rows and columns for train_df:-")
print(train_df.shape)
print("**************************************")
print("no. of rows and columns for test_df:-")
print(test_df.shape)
print("First five rows of train_df:-")
print(train_df.head())
print("************************************************************")
print("First five rows of test_df:-")
print(test_df.head())
print("*************************************************************")
print("Info for train_df:-")
print(train_df.info())
print("*************************************************************")
print("Info for test_df:-")
print(test_df.info())
print("None Values in train_df:-")
train_null=train_df.isnull().sum()
print(train_null)
print("**********************************")
print("None Values in test_df:-")
test_null=test_df.isnull().sum()
print(test_null)
print("null values in tes_df and train_df:-")
df = pd.DataFrame(
{"Item_Weight" : [train_null.Item_Weight,test_null.Item_Weight],
"Outlet_Size" : [train_null.Outlet_Size,test_null.Outlet_Size]},
 
index = ["train_df", "test_df"])
print(df)

#Creating a list of both datasets:-
full_data=[train_df,test_df]
print(train_df.columns)
print(train_df.info())


#print(train_df.Item_Fat_Content.value_counts())
#train_df.Item_Type.value_counts()
print(test_df.Outlet_Type.value_counts())
#train_df.Outlet_Identifier.value_counts()
test_item_identifier=test_df["Item_Identifier"]
test_outlet_identifier=test_df["Outlet_Identifier"]
train_item_identifier=train_df["Item_Identifier"]
for dataset in full_data:
    dataset['Item_Fat_Content']=dataset['Item_Fat_Content'].replace(['Low Fat','LF','low fat'],'Low')
    dataset['Item_Fat_Content']=dataset['Item_Fat_Content'].replace('reg','Regular')
    Item_Fat_mapping={"Low":1,"Regular":2}
    dataset['Item_Fat_Content']=dataset['Item_Fat_Content'].map(Item_Fat_mapping)
    
for dataset in full_data:
    Item_Type_mapping = {"Fruits and Vegetables": 1, "Snack Foods": 2, "Household": 3, "Frozen Foods": 4, "Dairy": 5,"Canned":6,"Baking Goods":7,"Health and Hygiene":8,"Soft Drinks":9,"Meat":10,"Breads":11,"Hard Drinks":12,"Others":13,"Starchy Foods":14,"Breakfast":15,"Seafood":16}
    dataset['Item_Type'] = dataset['Item_Type'].map(Item_Type_mapping)
    
for dataset in full_data:
    Outlet_Identifier_mapping={"OUT027":1,"OUT013":2,"OUT049":3,"OUT035":4,"OUT046":5,"OUT045":6,"OUT018":7,"OUT017":8,"OUT010":9,"OUT019":10}
    dataset['Outlet_Identifier']=dataset['Outlet_Identifier'].map(Outlet_Identifier_mapping)
    
for dataset in full_data:
    Outlet_Size_mapping={"High":1,"Medium":2,"Small":3}
    dataset['Outlet_Size']=dataset['Outlet_Size'].map(Outlet_Size_mapping)
    dataset['Outlet_Size']=dataset['Outlet_Size'].fillna(0)
       
for dataset in full_data:
    dataset['Item_Weight']=dataset['Item_Weight'].fillna(dataset.Item_Weight.mean())

for dataset in full_data:
    Outlet_Location_mapping={"Tier 1":1,"Tier 2":2,"Tier 3":3}
    dataset['Outlet_Location_Type']=dataset['Outlet_Location_Type'].map(Outlet_Location_mapping)

    
for dataset in full_data:
    Outlet_Type_mapping={"Supermarket Type1":1,"Supermarket Type2":2,"Supermarket Type3":3,"Grocery Store":4}
    dataset['Outlet_Type']=dataset['Outlet_Type'].map(Outlet_Type_mapping)
    
train_df.head(3)    
    
    
train_df=train_df.drop(['Item_Identifier'],axis=1)
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
y = train_df.Item_Outlet_Sales
Sales_Predictors = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 
                        'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type'
                  ]
X = train_df[Sales_Predictors]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)


print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))


val_X=test_df[Sales_Predictors]
sales_preds = my_model.predict(val_X)
sales_preds=pd.DataFrame(sales_preds)
sales_preds
sales_preds=sales_preds.rename( columns={0:"Item_Outlet_Sales"})
sales_preds
submission_file=pd.concat([test_outlet_identifier,sales_preds], axis=1)
submission_file=pd.concat([test_item_identifier,submission_file],axis=1)
submission_file
submission_file.to_csv('submission.csv', index=False)