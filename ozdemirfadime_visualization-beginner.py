# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from collections import Counter
%matplotlib inline
from wordcloud import WordCloud  # word cloud library
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data =pd.read_csv("../input/BlackFriday.csv")
data.head(10)
data.tail()
data.columns
data.info()
#how much  missing data and which features
import missingno as msno
msno.bar(data)
plt.show()
#only  missing data in product_category_1 and product_category_2 
#count of Null values
print(data.isnull().sum())
#Remove missing values.
data = data.dropna()

data.corr()
#Heat map corraliton of features
plt.figure(figsize =(15,15))
sns.heatmap(data.corr(),annot=True,fmt =".2f")
plt.show()
plt.figure(figsize=(15,14))
sns.barplot(x=data.Age, y=data.Purchase)
plt.xticks(rotation= 40)
plt.xlabel('Age')
plt.ylabel('Purchase')
plt.title('age to purchase')
plt.show()
#data['Purchase'].value_counts().head(10).plot.bar()
#data["Occupation"].value_counts().sort_index().plot.bar()

#Count Plot
sns.countplot(data.Occupation)
plt.title("Occupation",color = 'red',fontsize=18)
#female male ration 
sns.countplot(data.Gender)
plt.title("Gender",color="blue",fontsize =19)
#data.Age =data.Age.astype(float)
fig= plt.subplots(figsize=(12,7))
sns.countplot(data['Age'],hue=data['Gender'])
f,ax = plt.subplots(figsize = (10,15))
sns.barplot(x=data.Product_Category_1,y=data.Purchase,color='green',alpha = 0.5,label='product1' )
sns.barplot(x=data.Product_Category_2,y=data.Purchase,color='blue',alpha = 0.7,label='product2')
sns.barplot(x=data.Product_Category_3,y=data.Purchase,color="red",alpha =0.6,label="product3")
ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='productions', ylabel='Purchase',title = " productions of Purchase")
sns.despine(left=True, bottom=True)
#sns.set(style="ticks", palette="pastell")
#sns.despine(offset=30, trim=False)
sns.boxplot(x ="Gender",y ="Purchase",hue ='City_Category',data=data,palette="PRGn")
plt.show()
sns.jointplot(x=data.Product_Category_1,y =data.Purchase,kind ="kde",height =9,color="blue")
#data=data.Age.unique()
labels =data.City_Category.value_counts().index
explode = [0,0,0]
sizes = data.City_Category.value_counts().values
color =["red","green","blue"]
# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels,colors =color, autopct='%2.3f%%')
plt.title('city category of ration',fontsize = 17,color = 'Black')
#colors=sns.color_palette('Set2')


#line plot Production1 of Purchase 
#data['Purchase'].value_counts().sort_index().plot.line()
#data = sns.load_dataset("data")
#plt.style.use("seaborn-darkgrid")
sns.lineplot(x="Product_Category_1", y="Purchase",hue ="Age" ,data=data,markers=True)

data8=data["Occupation"]
#sns.violinplot(data=data8,inner="quart")
sns.violinplot(x="Gender",y="Occupation",data=data,inner="points")
data2 =data["Purchase"]
data3 =data["Occupation"]
data4=pd.concat([data2,data3],axis=1)
sns.pairplot(data4)
plt.show()