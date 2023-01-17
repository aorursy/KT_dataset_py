import numpy as np # Numerical Python
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline
from plotly.offline import iplot
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore') 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#I will analyze v2 because only this dataset includes the city feature
dataset=pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
dataset.head()
#General Info about dataset
dataset.info()
#Checking general statistical values
dataset.describe().T
city_quantity=dataset.city.value_counts()

plt.figure(figsize=(15,10))
plt.pie(x=city_quantity, labels=city_quantity.index, autopct='%1.1f%%')
plt.title('Ratio of Cities',color = 'red',fontsize = 35)
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(dataset['animal'])
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x="furniture", data=dataset)
plt.show()
f,ax = plt.subplots(figsize = (15,15))
sns.distplot(dataset["total (R$)"],bins=1000,kde=False)
plt.show()
#Let's check counts
print(dataset.floor.value_counts())
# '-' is not an acceptable value so i will drop that columns.
#Data cleaning
floor_data=dataset.drop(dataset[dataset.floor=='-'].index)
floor_data.floor=floor_data.floor.astype(int)

#Visualization Part
floor_data=floor_data.sort_values('floor')
floor_data.floor=floor_data.floor.astype(object)
plt.figure(figsize=(15,10))
sns.countplot(floor_data.floor)
plt.show()
plt.figure(figsize=(15,10))
sns.boxplot(x ='city',y='rent amount (R$)', data = dataset, showfliers = False)
plt.show()
#I will change room feature as a categoric feature in order to create a boxplot
room_data=dataset.iloc[:,[2,9]]
room_data['rooms']=room_data['rooms'].astype(object)
plt.figure(figsize=(15,10))
sns.boxplot(x ='rooms',y='rent amount (R$)', data = dataset)
plt.show()
#I will change bathroom feature as a categoric feature to create a boxplot
categoric_bathroom=dataset.copy()
categoric_bathroom.bathroom=categoric_bathroom.bathroom.astype(object)
plt.figure(figsize=(15,10))
sns.boxplot(x ='bathroom',y='rent amount (R$)', data = categoric_bathroom)
plt.show()
numeric_data=dataset.select_dtypes(include=['int64']).copy()
plt.figure(figsize=(15,10))
sns.pairplot(numeric_data)
plt.show()
#Correlation Heatmap
corelation_matrix=numeric_data.corr()
fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corelation_matrix, annot=True, linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax,cmap='jet')
plt.show()
sns.set(font_scale=1.5)
#I'm gonna use np.log() because there are many outliers that prevents a clear visualization
plt.figure(figsize=(15,10))
sns.scatterplot( x =np.log(dataset['area']+1) , y = np.log(dataset['rent amount (R$)']), 
                hue = dataset['bathroom'],size=dataset['parking spaces'],
                palette="afmhot",alpha=0.85)
plt.axis([2,8,6,10])
plt.show()
plt.figure(figsize=(15,10))
sns.lmplot(x='total (R$)',y='hoa (R$)',data=dataset)
plt.show()
plt.figure(figsize=(15,10))
sns.scatterplot(x='total (R$)',y='fire insurance (R$)',data=dataset)
plt.axis([0,35000,0,500])
plt.show()
import plotly.express as px
fig = px.scatter_3d(numeric_data, x='rooms',
                    y='bathroom', 
                    z=np.log(numeric_data['rent amount (R$)']), #I used np.log() again because there are many outliers. So z represents rent amount
                   color='rooms', 
       color_continuous_scale='icefire'
       )
iplot(fig)
#I will use a simple normalization technique in order to see all features in the same domain.
normalized_data=numeric_data.copy()
for column in normalized_data.columns:
    normalized_data[column]=normalized_data[column]/normalized_data[column].max()
    
normalized_data.head()
#Visualization part
fig,ax1 = plt.subplots(figsize =(15,9))
sns.pointplot(x=normalized_data['area'],y=normalized_data['rent amount (R$)'],data=normalized_data,color='lime',alpha=0.8)
sns.pointplot(x=normalized_data['bathroom'],y=normalized_data['rent amount (R$)'],data=normalized_data,color='red',alpha=0.8)
sns.pointplot(x=normalized_data['parking spaces'],y=normalized_data['rent amount (R$)'],data=normalized_data,color='darkslategray',alpha=0.6)
plt.xticks(rotation=90)
plt.text(5.5,0.50,'area-rent amount (R$)',color='red',fontsize = 18,style = 'italic')
plt.text(5.4,0.46,'rooms-price rent amount (R$)',color='lime',fontsize = 18,style = 'italic')
plt.text(5.3,0.42,'parking spaces-rent amount (R$)',color='darkslategray',fontsize = 18,style = 'italic')
plt.xlabel('X - Axis',fontsize = 15,color='blue')
plt.ylabel('Y - Axis',fontsize = 15,color='blue')
plt.title('Area-Rent amount (R$) vs Room-Rent amount (R$) vs Parking space-Rent amount (R$)',fontsize = 20,color='blue')
plt.grid()