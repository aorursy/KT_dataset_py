# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/insurance.csv')
data.info()
data.columns
data.corr()
f,ax = plt.subplots(figsize = (15,15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()
data.head(10)
data.age.plot(kind='line', color='g', label='Age', linewidth=1, alpha=0.5, grid=True, linestyle='-')
data.bmi.plot(kind='line', color='r', label='BMI', linewidth=1, alpha=0.5, grid=True, linestyle=':')
plt.legend('upper left')
plt.xlabel('age')
plt.ylabel('bmi')
plt.title('Line Plot')
plt.show()
data.plot(kind='scatter', x='age', y='children',alpha = 0.5,color = 'red')
plt.xlabel('Age')              
plt.ylabel('Children')
plt.title('Age Children Scatter Plot')  
plt.show()
data.charges.plot(kind='hist', bins=50, figsize=(10,10))
plt.show()
data1=data['sex']=='female'
data_female=data[data1]
data2=data['sex']=='male'
data_male=data[data2]
data_female.charges.plot(kind='hist', bins=50, figsize=(10,10))
plt.show()
data_male.charges.plot(kind='hist', bins=50, figsize=(10,10))
plt.show()
data4=(data['sex']=='female') & (data['smoker']=='yes') & (data['children']>0)
data[data4]
data3=(data['sex']=='male') & (data['smoker']=='yes') & (data['children']>0)
data[data3]
average_bmi=sum(data.bmi)/len(data.bmi)
data['bmi_level']=['high' if i>average_bmi else 'low' for i in data.bmi]
data.loc[:10,["bmi_level","bmi"]] # we will learn loc more detailed later

data.shape
data.tail()
print(data.region.value_counts(dropna = False))
data.describe()
data.boxplot(column='age',by = 'children')
data_new=data.head()
data_new
melted=pd.melt(frame=data_new,id_vars='age',value_vars=['children','bmi_level',])
melted
melted.pivot(index='age',columns='variable',values='value')
data_h=data.head()
data_t=data.tail()
conc_data_row=pd.concat([data_h,data_t],axis=0,ignore_index=True)
conc_data_row
data_1=data['region'].head()
data_2=data['smoker'].head()
conc_data_col=pd.concat([data_1,data_2],axis=1)
conc_data_col
data.dtypes
data['charges']=data['charges'].astype('category')
data['bmi_level']=data['bmi_level'].astype('object')
data.dtypes
assert 1==1
assert data.age.dtypes == np.int
data.plot(subplots=True)
plt.show()
data['charges']=data['charges'].astype('float')
data.plot(kind = "hist",y = "charges",bins = 50,range= (0,25000),normed = True)
plt.show()
fig, axes = plt.subplots(nrows=1,ncols=2)
data.plot(kind = "hist",y = "charges",bins = 150,range= (0,25000),normed = True,ax = axes[0])
data.plot(kind = "hist",y = "charges",bins = 150,range= (0,25000),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
time_list = ["1997-07-04","1995-10-05"]
print(type(time_list[1]))

datetime_object=pd.to_datetime(time_list)
print(type(datetime_object))
data5=data.head()
date_list=["1997-01-10","1997-02-10","1997-02-15","1997-03-10","1997-03-11"]
datetime_object = pd.to_datetime(date_list)
data5["date"] = datetime_object
data5=data5.set_index('date')
data5
print(data5.loc["1997-03-10"])
print(data5.loc["1997-02-10":"1997-03-16"])
data5.resample("A").mean()
data5.resample("M").mean()
data5.resample("M").mean().interpolate("linear")
data.loc[10:1:-1,"age":"children"]
boolean = (data.age<20) & (data.children>0) & (data.bmi>30)
data[boolean]
data.age[data.bmi<18]
print(data.index.name)
data.index.name="index_name"
data.head()
data_c=data.copy()
data_c.index=range(1,1339,1)
data_c.head()
data6 = data.set_index(["smoker","region"])
data6.head(50)
data.groupby("region").mean()