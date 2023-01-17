import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

import os
print(os.listdir("../input"))

data=pd.read_csv('../input/menu.csv')
data.columns
data.isnull().sum()
data.tail()
data.corr()
data.pivot_table('Protein','Category').plot(kind='bar',stacked=True,color='y')
data.pivot_table('Vitamin A (% Daily Value)','Category').plot(kind='bar',stacked=True,color='b')
f,ax=plt.subplots(figsize=(18,11))
sns.heatmap(data.corr(),annot=True,linewidths=5,fmt='.1f',ax=ax)
data.Calories.plot(kind='hist',bins=40,figsize=(18,11))
plt.xlabel('Calories')
plt.ylabel('Sugars')
#Histogram
data.info() #to remember what categories i have
#Scatter Plot
data.plot(kind='scatter',x='Cholesterol',y='Sugars',alpha=0.4,color='m')
plt.xlabel('Cholesterol')
plt.ylabel('Sugars')
plt.title('does it matter?')
data

data['Category']


dataFrame=data[['Calories']]
print(type(dataFrame))
print('')
series=data['Calories']
print(type(series))
data.shape #rows and columns
data.describe() #to remember what i ve
x=data['Sugars'] > 30
data[x]

data[(data['Calories']>300) & (data['Total Fat']>15)]
data[np.logical_and(data['Carbohydrates']>35, data['Calories']>400)]
data.loc[:5,"Carbohydrates"]
data.loc[:5, ["Category","Item"]]
data.loc[:3,"Item" :"Cholesterol"] #between Item and Cholesterol
threshold=sum(data['Calories'])/len(data['Calories'])
print('threshold is', threshold)

data["cal_level"]=["high" if i>threshold else "low" for i in data['Calories']]
data.loc[:15,["cal_level","Calories","Item"]] 
data['Calories'].value_counts().head(10).plot.bar()

data['Trans Fat'].value_counts().sort_index().plot.line()
data['Cholesterol'].value_counts().sort_index().plot.area()
data.plot.scatter(x='Carbohydrates (% Daily Value)',y='Carbohydrates')
data.info()
print(data['Item'].value_counts(dropna=False))
data.boxplot(column='Calories', by='Calcium (% Daily Value)')
data_new=data.head()
data_new
melted_data=pd.melt(frame=data_new,id_vars = 'Item',value_vars=['Calories','Protein'])
melted_data
melted_data.pivot(index='Item',columns='variable',values='value')
data1=data.head()
data2=data.tail()
conc_data=pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data
data3=data['Calories'].head()
data4=data['Calories from Fat'].head()
conc_data_col=pd.concat([data3,data4],axis=1)
conc_data_col
data.dtypes
data['Cholesterol']=data['Cholesterol'].astype('float64')
data['Serving Size']=data['Serving Size'].astype('category')
data.dtypes
data5=data.loc[:,["Calcium (% Daily Value)","Vitamin A (% Daily Value)","Vitamin C (% Daily Value)"]]
data5.plot()
data5.plot(subplots=True)
plt.show()
data.plot(kind="scatter",x="Calories",y="Vitamin C (% Daily Value)")
plt.show()
data5.plot(kind='hist',y='Calcium (% Daily Value)',bins=50,range=(0,160),normed=True)
# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data.plot(kind = "hist",y = "Calories",bins = 50,range= (0,250),normed = True,ax = axes[0])
data.plot(kind = "hist",y = "Total Fat (% Daily Value)",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt







