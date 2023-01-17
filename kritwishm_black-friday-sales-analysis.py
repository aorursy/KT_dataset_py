import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv('../input/BlackFriday.csv')
train.head()
train.info()
train.isnull().sum()
sns.set_style(style='whitegrid')
sns.countplot(x='Gender',data=train)
sns.countplot(x='Gender',data=train,hue='Marital_Status',palette='coolwarm')
sns.countplot(x='Age',data=train,hue='Gender')
train['combined_G_M']=train.apply(lambda x:'%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)
sns.countplot(x='Age',data=train,hue='combined_G_M')
train.fillna(value=0,inplace=True)
train['Product_Category_2']=train['Product_Category_2'].astype(int)
train['Product_Category_3']=train['Product_Category_3'].astype(int)
train.drop(['User_ID','Product_ID'],axis=1,inplace=True)
sns.countplot(x='Product_Category_2',data=train,hue='combined_G_M')
plt.figure(figsize=(12,10))
sns.countplot(x='Product_Category_3',data=train,hue='combined_G_M')
corrmat=train.corr()
fig,ax=plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=.8,square=True)
train.head()
def map_gender(gender):
    if gender == 'M':
        return 1
    else:
        return 0
train['Gender'] = train['Gender'].apply(map_gender)
def map_age(age):
    if age == '0-17':
        return 0
    elif age == '18-25':
        return 1
    elif age == '26-35':
        return 2
    elif age == '36-45':
        return 3
    elif age == '46-50':
        return 4
    elif age == '51-55':
        return 5
    else:
        return 6

train['Age'] = train['Age'].apply(map_age)
def map_city_categories(city_category):
    if city_category == 'A':
        return 2
    elif city_category == 'B':
        return 1
    else:
        return 0
train['City_Category']=train['City_Category'].apply(map_city_categories)
def map_stay(stay):
    if stay == '4+':
        return 4
    else:
        return int(stay)
train['Stay_In_Current_City_Years']=train['Stay_In_Current_City_Years'].apply(map_stay)
train.head()
corrmat=train.corr()
plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=0.8,square=True)