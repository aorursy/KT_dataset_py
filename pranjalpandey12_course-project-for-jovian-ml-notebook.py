project_name = "Project-Work-Zero-to-Pandas" 
!pip install jovian --upgrade -q
# Loading required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("../input/big-mart-sales-prediction/Train.csv")
data.head(5)
data.shape
data.columns
data.isnull().sum()
# Filling missing values in Item_Weight by mean 
data["Item_Weight"].fillna(np.nanmean(data["Item_Weight"]) , inplace = True)
# Frequency table for Outlet_Size
data["Outlet_Size"].value_counts()
# Filling missing values in Outlet_Size by most frequent value
data["Outlet_Size"].fillna("Medium" , inplace = True)
# Again checking the presence of missing values
data.isnull().sum()
# Creating a column named "MRP per weight"
data["MRP per weight"] = data["Item_MRP"]/data["Item_Weight"]
# Displaying first five rows and all columns
data.head(5)
# Frequency table for Item_Type
data["Item_Type"].value_counts()
# Frequency table for Outlet_Location_Type
data["Outlet_Location_Type"].value_counts()
# Frequency table for Outlet_Type
data["Outlet_Type"].value_counts()
# Frequency table for Item_Fat_Content
data["Item_Fat_Content"].value_counts()
data['Item_Fat_Content'] = np.where((data.Item_Fat_Content == 'LF'),'Low Fat',data.Item_Fat_Content)
data['Item_Fat_Content'] = np.where((data.Item_Fat_Content == 'low fat'),'Low Fat',data.Item_Fat_Content)
data['Item_Fat_Content'] = np.where((data.Item_Fat_Content == 'reg'),'Regular Fat',data.Item_Fat_Content)
data['Item_Fat_Content'] = np.where((data.Item_Fat_Content == 'Regular'),'Regular Fat',data.Item_Fat_Content)
# Again displaying frequency table for Item_Fat_Content
data["Item_Fat_Content"].value_counts()
# Determinig five point summary measures for all numeric varibles
data.describe(include = ['float'] , exclude = ['int'])
# Distribution of Item_MRP
sns.distplot(data['Item_MRP'], hist=True, kde=True, 
             bins=int(180/2), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2})
# Distribution of MRP per weight
sns.distplot(data['MRP per weight'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2})
# Distribution of Item_Outlet_Sales
sns.distplot(data['Item_Outlet_Sales'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2})
# Scatter plot between Item_Weight and Item_MRP
plt.scatter(data['Item_Weight'] , data['Item_MRP'])
# Scatter plot between Item_Weight and Item_Outlet_Sales
plt.scatter(data['Item_Weight'] , data['Item_Outlet_Sales'])
# Scatter plot between Item_MRP and Item_Outlet_Sales
plt.scatter(data['Item_MRP'] , data['Item_Outlet_Sales'])
# Scatter plot between MRP per weight and Item_Outlet_Sales
plt.scatter(data['MRP per weight'] , data['Item_Outlet_Sales'])
plt.scatter(data['Item_Weight'] , data['Item_MRP'])
sns.boxplot(data['Item_Fat_Content'], data['Item_MRP'])
plt.figure(figsize=(25, 6))
sns.boxplot(data['Item_Type'], data['Item_MRP'])
plt.figure(figsize=(14,6))
sns.barplot(data['Outlet_Type'],data['Item_Outlet_Sales'])
plt.figure(figsize=(14,6))
sns.barplot(data['Outlet_Size'],data['Item_Outlet_Sales'])
data.groupby(['Outlet_Type','Outlet_Size'])['Item_Outlet_Sales'].count()
plt.figure(figsize=(14,6))
sns.boxplot(data['Outlet_Size'], data['Item_Outlet_Sales'])
plt.figure(figsize=(14,6))
sns.boxplot(data['Outlet_Type'], data['Item_Outlet_Sales'])
data.groupby(['Item_Fat_Content','Outlet_Type'])['Item_Outlet_Sales'].count()
data.groupby(['Item_Fat_Content','Outlet_Type'])['Item_Outlet_Sales'].count()
import jovian
jovian.commit(project = project_name , environment=None)