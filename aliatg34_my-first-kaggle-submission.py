import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn  as sns
%matplotlib inline
from collections import Counter
import os
print(os.listdir("../input"))
import plotly.plotly as py
data=pd.read_csv('../input/BlackFriday.csv')
data.info()

data.corr()
#correlation map
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), linecolor="blue", linewidths=0.5, annot=True, fmt=".1f")
plt.show()
data.head()
explode = (0.1,0)  
fg, ax = plt.subplots(figsize=(12,8))
ax.pie(data['Gender'].value_counts(), explode=explode,labels=['Male','Female'], autopct='%1.1f%%',
        shadow=True, startangle=180)
# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()

data.plot(kind='scatter', x='Occupation', y='Product_Category_1',alpha = 0.01,color = 'blue')
plt.xlabel('Occupation')             
plt.ylabel('Product_Category_1')
plt.title('Occupation Product_Category_1 Scatter Plot') 
data.columns
def plot(group,column,plot):
    ax=plt.figure(figsize=(12,6))
    data.groupby(group)[column].sum().sort_values().plot(plot)
plot('Gender','Purchase','bar')

fig, ax = plt.subplots(figsize=(12,7))
sns.countplot(y=data['Age'],hue=data['Gender'])
plot('Age','Purchase','bar')
explode = (0.1, 0, 0)
fig, ax = plt.subplots(figsize=(12,7))
ax.pie(data['City_Category'].value_counts(),explode=explode, labels=data['City_Category'].unique(), autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()
explode = (0.1, 0, 0)
fig, ax = plt.subplots(figsize=(12,7))
ax.pie(data.groupby('City_Category')['Purchase'].sum(),explode=explode, labels=data['City_Category'].unique(), autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()
fig, ax = plt.subplots(figsize=(12,7))
sns.countplot(data['City_Category'],hue=data['Age'])
#label=['Underage 0-17','Retired +55','Middleage 26-35','46-50 y/o','Oldman 51-55','Middleage+ 36-45','Youth']
explode = (0.1, 0)
fig, ax = plt.subplots(figsize=(12,7))
ax.pie(data['Marital_Status'].value_counts(),explode=explode, labels=['Yes','No'], autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()
data.columns
labels=['First Year','Second Year','Third Year','More Than Four Years','Geust']
explode = (0.1, 0.1,0,0,0)
fig, ax = plt.subplots(figsize=(12,7))
ax.pie(data.groupby('Stay_In_Current_City_Years')['Purchase'].sum(),explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()
labels=['First Year','Second Year','Third Year','More Than Four Years','Geust']
explode = (0.1, 0.1,0,0,0)
fig, ax = plt.subplots(figsize=(12,7))
ax.pie(data['Stay_In_Current_City_Years'].value_counts(),explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()
plot('Stay_In_Current_City_Years','Purchase','bar')
fig, ax = plt.subplots(figsize=(12,7))
data['Occupation'].value_counts().sort_values().plot('bar')
plot('Occupation','Purchase','bar')
plot('Product_Category_1','Purchase','barh')
plot('Product_Category_2','Purchase','barh')
plot('Product_Category_3','Purchase','barh')
fig, ax = plt.subplots(figsize=(12,7))
data.groupby('Product_ID')['Purchase'].count().nlargest(10).sort_values().plot('barh')
fig, ax = plt.subplots(figsize=(12,7))
data.groupby('Product_ID')['Purchase'].sum().nlargest(10).sort_values().plot('barh')