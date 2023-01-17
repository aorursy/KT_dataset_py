# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


import matplotlib.pyplot as plt
%matplotlib inline
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df= pd.read_csv('../input/BlackFriday.csv')

df.head()
print('<Contain NaNs?>')
print(df.isnull().any())
missing_ser_percentage = (df.isnull().sum()/df.shape[0]*100).sort_values(ascending=False)
missing_ser_percentage = missing_ser_percentage[missing_ser_percentage!=0].round(2)
missing_ser_percentage.name = 'missing values %'
print('\n<NaN ratio>')
print(missing_ser_percentage)
df.fillna(0,inplace=True)

for col in df.columns:
    print('{} unique element: {}'.format(col,df[col].nunique()))
df.dtypes

#unique values in Gender parameter
gender = np.unique(df['Gender'])
gender
def map_gender(gender):
    if gender == 'M':
        return 0
    else:
        return 1
df['Gender'] = df['Gender'].apply(map_gender)
age = np.unique(df['Age'])
age

city_category = np.unique(df['City_Category'])
city_category
def map_city_categories(city_category):
    if city_category == 'A':
        return 2
    elif city_category == 'B':
        return 1
    else:
        return 0
df['City_Category'] = df['City_Category'].apply(map_city_categories)
city_stay = np.unique(df['Stay_In_Current_City_Years'])
city_stay
df.head()

ageData = sorted(list(zip(df.Age.value_counts().index, df.Age.value_counts().values)))
age, productBuy = zip(*ageData)
age, productBuy = list(age), list(productBuy)
ageSeries = pd.Series((i for i in age))

data = [go.Bar(x=age, 
               y=productBuy, 
               name="How many products were sold",
               marker = dict(color=['#00FFFF', '#0000A0', '#ADD8E6', '#C0C0C0', '#808080', '#A52A2A', '#FF0000'],
                            line = dict(color='#7C7C7C', width = .5)),
              text="Age: " + ageSeries)]
layout = go.Layout(title= "How many products were sold by ages")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
age0_17 = len(df[df.Age == "0-17"].User_ID.unique())
age18_25 = len(df[df.Age == "18-25"].User_ID.unique())
age26_35 = len(df[df.Age == "26-35"].User_ID.unique())
age36_45 = len(df[df.Age == "36-45"].User_ID.unique())
age46_50 = len(df[df.Age == "46-50"].User_ID.unique())
age51_55 = len(df[df.Age == "51-55"].User_ID.unique())
age55 = len(df[df.Age == "55+"].User_ID.unique())
agesBuyerCount = [age0_17,age18_25,age26_35,age36_45,age46_50,age51_55,age55]
               
trace1 = go.Bar(x = age,
                y = agesBuyerCount,
                name = "People count",
                marker = dict(color=['#F3B396', '#F3F196', '#A7F9AD', '#D5F0EF', '#AAADEE', '#EAC1E8', '#DF8787'],
                             line = dict(color='#7C7C7C', width = 1)),
                text = "Age: " + ageSeries)
data = [trace1]
layout = go.Layout(title= "How many people did shopping by ages")
fig = go.Figure(data=data, layout=layout)
iplot(fig)

plt.figure(figsize=(15,5))
age_order = ['0-17','18-25','26-35','36-45','46-50','51-55','55+']
plt.subplot(131)
sns.countplot('Age',order=age_order,hue='Gender',data=df,alpha = 0.8)
plt.xlabel('Age',fontsize=14)
plt.ylabel('')
plt.xticks(rotation=70)
plt.title('Number of customers',fontsize=14)
plt.legend(['Male','Female'],frameon=True,fontsize=14)
plt.tick_params(labelsize=15)
plt.subplot(132)
df_Tpurchase_by_Age = df.groupby(['Age','Gender']).agg({'Purchase':np.sum}).reset_index()
sns.barplot('Age','Purchase',hue='Gender',data=df_Tpurchase_by_Age,alpha = 0.8)
plt.xlabel('Age',fontsize=14)
plt.ylabel('')
plt.xticks(rotation=70)
plt.title('Total purchase',fontsize=14)
plt.legend().set_visible(False)
plt.tick_params(labelsize=15)
plt.subplot(133)
df_Apurchase_by_Age = df.groupby(['Age','Gender']).agg({'Purchase':np.mean}).reset_index()
sns.barplot('Age','Purchase',hue='Gender',data=df_Apurchase_by_Age,alpha = 0.8)
plt.xlabel('Age',fontsize=14)
plt.ylabel('')
plt.xticks(rotation=70)
plt.title('Average purchase',fontsize=14)
plt.legend().set_visible(False)
plt.tick_params(labelsize=15)
x_Gender = ['Male', 'Female']
y_PurchaseAmountAccordingToGender = [df[df.Gender == 0 ].Purchase.sum(), df[df.Gender == 1].Purchase.sum()]

data = [go.Bar(x = x_Gender, 
                y = y_PurchaseAmountAccordingToGender,
              marker = dict(color=['#A7F9AD','#AAADEE']))]
layout = go.Layout(title = 'Purchased Amount According To Gender (in US Dollars)')
fig = go.Figure(data=data, layout=layout)
iplot(fig)
x_MaritalStatus = ['Single', 'Married']
y_PurchaseAmountAccordingToMaritalStatus = [int(df[df.Marital_Status == 0].Purchase.sum()), int(df[df.Marital_Status == 1].Purchase.sum())]

data = [go.Bar(x = x_MaritalStatus, 
                y = y_PurchaseAmountAccordingToMaritalStatus,
              marker = dict(color=['#D5F0EF','#AAADEE']))]
layout = go.Layout(title = 'Purchased Amount According To Marital Status (in US Dollars)')
fig = go.Figure(data=data, layout=layout)
iplot(fig)
x_Status = ['Single & Male', 'Single & Female', 'Married & Male', 'Married & Female']
y_Purchases = [df[(df.Gender == 0) & (df.Marital_Status == 0)].Purchase.sum(),
              df[(df.Gender == 1) & (df.Marital_Status == 0)].Purchase.sum(),
              df[(df.Gender == 0) & (df.Marital_Status == 1)].Purchase.sum(),
              df[(df.Gender == 1) & (df.Marital_Status == 1)].Purchase.sum()]

data = [go.Bar(x = x_Status, 
                y = y_Purchases,
              marker = dict(color=['#F3B396','#EAC1E8','#F3F196','#AAADEE']))]
layout = go.Layout(title = 'Purchased Amount According To Gender and Marital Status (in US Dollars)')
fig = go.Figure(data=data, layout=layout)
iplot(fig)
x_CityC = ['A','B','C']
y_PurchaseAmountAccordingToCity = [df[df.City_Category == 2].Purchase.sum(), df[df.City_Category == 1].Purchase.sum(), df[df.City_Category == 0].Purchase.sum()]
data = [go.Bar(x = x_CityC, 
                y = y_PurchaseAmountAccordingToCity,
              marker = dict(color=['#F3F196','#AAADEE','#F3B396']))]
layout = go.Layout(title = 'Purchased Amount According To City Category (in US Dollars)')
fig = go.Figure(data=data, layout=layout)
iplot(fig)
city_order = ['A','B','C']
plt.figure(figsize=(15,5))
plt.subplot(131)
df_Tpurchase_by_City = df.groupby(['City_Category','Occupation']).agg({'Purchase':np.sum}).reset_index()
sns.barplot('City_Category','Purchase',hue='Occupation',data=df_Tpurchase_by_City,alpha = 0.8)
plt.title('Total purchase',fontsize=14)
plt.xlabel('City',fontsize=14)
plt.ylabel('')
plt.legend().set_visible(False)
plt.tick_params(labelsize=15)
plt.subplot(132)
df_Apurchase_by_City = df.groupby(['City_Category','Occupation']).agg({'Purchase':np.mean}).reset_index()
sns.barplot('City_Category','Purchase',hue='Occupation',data=df_Apurchase_by_City,alpha = 0.8)
plt.title('Average purchase',fontsize=14)
plt.xlabel('City',fontsize=14)
plt.ylabel('')
plt.legend(title='Occupation',frameon=True,fontsize=10,bbox_to_anchor=(1,0.5), loc="center left")
plt.tick_params(labelsize=15)
labels = sorted(df.Stay_In_Current_City_Years.unique())
values = df.Stay_In_Current_City_Years.value_counts().sort_index()

trace = go.Pie(labels=labels, values=values)

iplot([trace])
df['Marital_Status_label']=np.where(df['Marital_Status'] == 0,'Single','Married')
df_Tpurchase_by_City_Marital = df.groupby(['City_Category','Marital_Status_label']).agg({'Purchase':np.sum}).reset_index()
df_Tpurchase_by_City_Stay = df.groupby(['City_Category','Stay_In_Current_City_Years']).agg({'Purchase':np.sum}).reset_index()
fig = plt.figure(figsize=(15,5))
fig.suptitle('Total purchase',fontsize=20)
plt.subplot(121)
sns.barplot('City_Category','Purchase',hue='Marital_Status_label',data=df_Tpurchase_by_City_Marital,alpha = 0.8)
plt.xlabel('City',fontsize=14)
plt.ylabel('')
plt.legend(frameon=True,fontsize=14)
plt.tick_params(labelsize=15)
plt.subplot(122)
sns.barplot('City_Category','Purchase',hue='Stay_In_Current_City_Years',data=df_Tpurchase_by_City_Stay,alpha = 0.8)
plt.xlabel('City',fontsize=14)
plt.ylabel('')
plt.legend(title='Residency duration',frameon=True,fontsize=12,loc=2)
plt.tick_params(labelsize=15)
df.drop('Marital_Status_label',axis=1,inplace=True)
corrmat = df.corr()
fig,ax = plt.subplots(figsize = (15,9))
sns.heatmap(corrmat, vmax=.8, square=True)
mean_cat_1 = df['Product_Category_1'].mean()
mean_cat_2 = df['Product_Category_2'].mean()
mean_cat_3= df['Product_Category_3'].mean()
print(f"PC1: {mean_cat_1} \n PC2: {mean_cat_2} \n PC3 : {mean_cat_3}")
from sklearn.preprocessing import LabelEncoder
df_Tpurchase_by_PC1_Age = df.groupby(['Product_Category_3','Age']).agg({'Purchase':np.sum}).reset_index()
fig = plt.figure(figsize=(12,5))
plt.subplot(121)
sns.countplot('Product_Category_3',hue='Age',data=df,alpha = 0.8,hue_order=age_order)
plt.title('Item count',fontsize=14)
plt.xlabel('Product category 3',fontsize=14)
plt.ylabel('')
plt.legend(title='Age group',frameon=True,fontsize=12)
plt.tick_params(labelsize=15)
plt.subplot(122)
sns.barplot('Product_Category_3','Purchase',hue='Age',data=df_Tpurchase_by_PC1_Age,alpha = 0.8)
plt.title('Total purchase',fontsize=14)
plt.xlabel('Product category 3',fontsize=14)
plt.ylabel('')
plt.legend().set_visible(False)
plt.tick_params(labelsize=15)

df_Tpurchase_by_PC1_Gender = df.groupby(['Product_Category_3','Gender']).agg({'Purchase':np.sum}).reset_index()
fig = plt.figure(figsize=(12,5))
plt.subplot(121)
sns.countplot('Product_Category_3',hue='Gender',data=df,alpha = 0.8)
plt.title('Item count',fontsize=14)
plt.xlabel('Product category 3',fontsize=14)
plt.ylabel('')
plt.legend(['Male','Female'],frameon=True,fontsize=12)
plt.tick_params(labelsize=15)
plt.subplot(122)
sns.barplot('Product_Category_3','Purchase',hue='Gender',data=df_Tpurchase_by_PC1_Gender,alpha = 0.8)
plt.title('Total purchase',fontsize=14)
plt.xlabel('Product category 3',fontsize=14)
plt.ylabel('')
plt.legend().set_visible(False)
plt.tick_params(labelsize=15) 
df_Tpurchase_by_PC1_Age = df.groupby(['Product_Category_1','Age']).agg({'Purchase':np.sum}).reset_index()
fig = plt.figure(figsize=(12,5))
plt.subplot(121)
sns.countplot('Product_Category_1',hue='Age',data=df,alpha = 0.8,hue_order=age_order)
plt.title('Item count',fontsize=14)
plt.xlabel('Product category 1',fontsize=14)
plt.ylabel('')
plt.legend(title='Age group',frameon=True,fontsize=12)
plt.tick_params(labelsize=15)
plt.subplot(122)
sns.barplot('Product_Category_1','Purchase',hue='Age',data=df_Tpurchase_by_PC1_Age,alpha = 0.8)
plt.title('Total purchase',fontsize=14)
plt.xlabel('Product category 1',fontsize=14)
plt.ylabel('')
plt.legend().set_visible(False)
plt.tick_params(labelsize=15)

df_Tpurchase_by_PC1_Gender = df.groupby(['Product_Category_1','Gender']).agg({'Purchase':np.sum}).reset_index()
fig = plt.figure(figsize=(12,5))
plt.subplot(121)
sns.countplot('Product_Category_1',hue='Gender',data=df,alpha = 0.8)
plt.title('Item count',fontsize=14)
plt.xlabel('Product category 1',fontsize=14)
plt.ylabel('')
plt.legend(['Male','Female'],frameon=True,fontsize=12)
plt.tick_params(labelsize=15)
plt.subplot(122)
sns.barplot('Product_Category_1','Purchase',hue='Gender',data=df_Tpurchase_by_PC1_Gender,alpha = 0.8)
plt.title('Total purchase',fontsize=14)
plt.xlabel('Product category 1',fontsize=14)
plt.ylabel('')
plt.legend().set_visible(False)
plt.tick_params(labelsize=15)
df_Tpurchase_by_PC1_Age = df.groupby(['Product_Category_2','Age']).agg({'Purchase':np.sum}).reset_index()
fig = plt.figure(figsize=(12,5))
plt.subplot(121)
sns.countplot('Product_Category_2',hue='Age',data=df,alpha = 0.8,hue_order=age_order)
plt.title('Item count',fontsize=14)
plt.xlabel('Product category 2',fontsize=14)
plt.ylabel('')
plt.legend(title='Age group',frameon=True,fontsize=12)
plt.tick_params(labelsize=15)
plt.subplot(122)
sns.barplot('Product_Category_2','Purchase',hue='Age',data=df_Tpurchase_by_PC1_Age,alpha = 0.8)
plt.title('Total purchase',fontsize=14)
plt.xlabel('Product category 2',fontsize=14)
plt.ylabel('')
plt.legend().set_visible(False)
plt.tick_params(labelsize=15)

df_Tpurchase_by_PC1_Gender = df.groupby(['Product_Category_2','Gender']).agg({'Purchase':np.sum}).reset_index()
fig = plt.figure(figsize=(12,5))
plt.subplot(121)
sns.countplot('Product_Category_2',hue='Gender',data=df,alpha = 0.8)
plt.title('Item count',fontsize=14)
plt.xlabel('Product category 2',fontsize=14)
plt.ylabel('')
plt.legend(['Male','Female'],frameon=True,fontsize=12)
plt.tick_params(labelsize=15)
plt.subplot(122)
sns.barplot('Product_Category_2','Purchase',hue='Gender',data=df_Tpurchase_by_PC1_Gender,alpha = 0.8)
plt.title('Total purchase',fontsize=14)
plt.xlabel('Product category 2',fontsize=14)
plt.ylabel('')
plt.legend().set_visible(False)
plt.tick_params(labelsize=15)
