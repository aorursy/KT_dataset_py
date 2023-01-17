#Importing the required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Reading the train data

data_train = pd.read_csv("../input/big-mart-sales-dataset/Train_UWu5bXk.csv")

#Visualizing data is very important as it provides a better idea about the features of the dataset and how it influences the target variable
#This is a line plot to show the variation of the target variable across the dataset
import seaborn as sns
sns.lineplot(data = data_train['Item_Outlet_Sales'])

#This scatter plot show the variation of Item MRP vs Item Sales using scatter plot
sns.scatterplot(x = data_train['Item_MRP'],y=data_train['Item_Outlet_Sales'])
#This shows the variation of Item Fat Content vs Item Sales using bar plot
sns.barplot(x = data_train['Item_Fat_Content'],y = data_train['Item_Outlet_Sales'])
#This is a variation of outletlocation type vs sales using a swarm plot
sns.swarmplot(x=data_train['Outlet_Location_Type'],y=data_train['Item_Outlet_Sales'])
sns.distplot(a = data_train['Item_Outlet_Sales'])

sns.kdeplot(data = data_train['Item_Outlet_Sales'], shade = True)
#A simple joint plot to visualize item MRP and outlet sales
sns.jointplot(x = data_train['Item_Weight'],y = data_train['Item_Outlet_Sales'])