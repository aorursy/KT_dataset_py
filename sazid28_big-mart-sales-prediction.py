import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
import warnings
warnings.filterwarnings("ignore")
train_data=pd.read_csv("../input/Train .csv")
test_data = pd.read_csv("../input/Test .csv")
train_data.head(5)
test_data.head(5)
train_original = train_data.copy()
test_original = test_data.copy()
train_original.head(2)
test_original.head(2)
train_data.shape , train_original.shape
test_data.shape , test_original.shape
train_data.info()
train_data.columns
train_data.dtypes
train_data.isnull().sum()
test_data.isnull().sum()
full_data = [train_data,test_data]
train_data.columns
test_data.columns
train_data["Item_Fat_Content"].value_counts()
test_data['Item_Fat_Content'].value_counts()
for dataset in full_data:
    dataset['Item_Fat_Content']=dataset['Item_Fat_Content'].replace(['Low Fat','LF','low fat'],'Low')
    dataset['Item_Fat_Content']=dataset['Item_Fat_Content'].replace('reg','Regular')
    
train_data['Item_Fat_Content'].value_counts()
test_data['Item_Fat_Content'].value_counts()
for dataset in full_data:
    Item_Fat_mapping={"Low":1,"Regular":2}
    dataset['Item_Fat_Content']=dataset['Item_Fat_Content'].map(Item_Fat_mapping)
train_data['Item_Fat_Content'].value_counts()
test_data['Item_Fat_Content'].value_counts()
train_data['Item_Fat_Content'].value_counts(normalize=True).plot.bar(title='Item_Fat_Content')
train_data['Item_Fat_Content'].value_counts(normalize=True)*100
train_data['Item_Type'].value_counts()
for dataset in full_data:
    Item_Type_mapping={'Fruits and Vegetables':1,'Snack Foods':2,'Household':3,'Frozen Foods':4,'Dairy':5,'Canned':6, 
    'Baking Goods':7,'Health and Hygiene':8,'Soft Drinks':9,'Meat':10,'Breads':11,'Hard Drinks':12,'Others':13,'Starchy Foods':14,
    'Breakfast':15,'Seafood':16}
    dataset['Item_Type']=dataset['Item_Type'].map(Item_Type_mapping)
train_data['Item_Type'].value_counts()
test_data['Item_Type'].value_counts() #just compare with train set
train_data['Item_Type'].value_counts(normalize=True).plot.bar(title='Item_Type')
train_data['Item_Type'].value_counts(normalize=True)*100  #calculate the value in percentage
train_data['Outlet_Identifier'].value_counts()
test_data['Outlet_Identifier'].value_counts() #just compare with train set
for dataset in full_data:
    Outlet_Identifier_mapping = {'OUT027':1,'OUT013':2,'OUT046':3,'OUT049':4,'OUT035':5,'OUT045':6,'OUT018':7,'OUT017':8,
    'OUT010':9,'OUT019':10}
    dataset['Outlet_Identifier']=dataset['Outlet_Identifier'].map(Outlet_Identifier_mapping)
train_data['Outlet_Identifier'].value_counts()
test_data['Outlet_Identifier'].value_counts()
train_data['Outlet_Identifier'].value_counts(normalize=True).plot.bar(title='Outlet_Identifier')
train_data['Outlet_Identifier'].value_counts(normalize=True)*100
train_data['Outlet_Size'].value_counts()
test_data['Outlet_Size'].value_counts()
for dataset in full_data:
    Outlet_Size_mapping={'Small':1,'Medium':2,'High':3}
    dataset['Outlet_Size']=dataset['Outlet_Size'].map(Outlet_Size_mapping)
    dataset['Outlet_Size']=dataset['Outlet_Size'].fillna(4)
train_data['Outlet_Size'].value_counts()
test_data['Outlet_Size'].value_counts()
train_data['Outlet_Size'].value_counts(normalize=True).plot.bar(title='Outlet_Size')
train_data['Outlet_Size'].value_counts(normalize=True)*100
train_data['Outlet_Location_Type'].value_counts()
test_data['Outlet_Location_Type'].value_counts()
for dataset in full_data:
    Outlet_Location_Type_mapping={'Tier 1':1,'Tier 2':2,'Tier 3':3}
    dataset['Outlet_Location_Type']=dataset['Outlet_Location_Type'].map(Outlet_Location_Type_mapping)
train_data['Outlet_Location_Type'].value_counts()
train_data['Outlet_Location_Type'].value_counts(normalize=True).plot.bar(title='Outlet_Location_Type')
train_data['Outlet_Location_Type'].value_counts(normalize=True)*100
train_data['Outlet_Type'].value_counts()
test_data['Outlet_Type'].value_counts()
for dataset in full_data:
    Outlet_Type_mapping={'Supermarket Type1':1,'Supermarket Type2':2,'Supermarket Type3':3,'Grocery Store':4}
    dataset['Outlet_Type']=dataset['Outlet_Type'].map(Outlet_Type_mapping)
train_data['Outlet_Type'].value_counts()
test_data['Outlet_Type'].value_counts()
train_data['Outlet_Type'].value_counts(normalize=True).plot.bar(title='Outlet_Size')
train_data['Outlet_Type'].value_counts(normalize=True)*100
for dataset in full_data:
    dataset['Item_Weight']=dataset['Item_Weight'].fillna(dataset.Item_Weight.mean)
train_data['Item_Weight'].isnull().sum()
test_data['Item_Weight'].isnull().sum()
train_data.isnull().sum()
test_data.isnull().sum()
train_data.head(5)
test_data.head(5)
#plt.figure(figsize=(10,6))
#sns.heatmap(train_data.corr(),annot=True)
matrix = train_data.corr()
f, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu",annot=True);
sns.distplot(train_data["Item_Outlet_Sales"])
train_data["Item_Outlet_Sales"].hist(bins=20)
train_data["Item_Outlet_Sales_log"]=np.log(train_data["Item_Outlet_Sales"])
train_data["Item_Outlet_Sales_log"].hist(bins=20)
sns.distplot(train_data["Item_Outlet_Sales_log"])
train_data.head(3)
X.head(3)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)
X.drop["Outlet_Establishment_Year"]
X.head()
