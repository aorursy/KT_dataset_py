# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
audi=pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/audi.csv')
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
matplotlib.axes.Axes.pie
matplotlib.pyplot.pie
matplotlib.axes.Axes.legend
matplotlib.pyplot.legend
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
audi.head(5)
audi.shape
audi.isnull().sum()
#Pie Plot and Count Plot
audi['model'].unique().shape
x=np.zeros((26,))
for i in range(1,10,1):
  x[i]=0.1

f,ax=plt.subplots(1,2,figsize=(22,8))
audi['model'].value_counts().plot.pie(explode=x,autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Model')
sns.countplot(audi['model'],ax=ax[1])
ax[1].set_title('Model')
plt.show()
audi.columns
audi.groupby(['model','year'])['year'].count()
plot=sns.countplot('year',data=audi)
plot.set_xticklabels(plot.get_xticklabels(), rotation=90)

plt.show()

f,ax=plt.subplots(1,2,figsize=(18,8))
audi['transmission'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_ylabel('Count')
sns.countplot('transmission',hue='fuelType',data=audi,ax=ax[1])
plt.show()
sns.factorplot('year','transmission',hue='fuelType',data=audi)
plt.show()
pd.crosstab([audi.year,audi.model],[audi.transmission,audi.fuelType],margins=True).style.background_gradient(cmap='summer_r')

sns.factorplot('model','year',data=audi,)
fig=plt.gcf()
fig.set_size_inches(10,3)
plt.show()
sns.factorplot('fuelType','price',col='transmission',data=audi)
fig=plt.gcf()
fig.set_size_inches(20,8)
plt.show()
f,ax=plt.subplots(3,figsize=(20,8))
sns.distplot(audi[audi['transmission']=='Manual'].price,ax=ax[0])
ax[0].set_title('Price in Manual Audi')
sns.distplot(audi[audi['transmission']=='Semi-Auto'].price,ax=ax[1])
ax[1].set_title('Price in Semi-Auto Audi')
sns.distplot(audi[audi['transmission']=='Automatic'].price,ax=ax[2])
ax[2].set_title('Price in Automatic Audi')
plt.show()
f,ax=plt.subplots(3,figsize=(20,8))
sns.distplot(audi[audi['transmission']=='Manual'].price,ax=ax[0])
ax[0].set_title('Price in Petrol Audi')
sns.distplot(audi[audi['transmission']=='Semi-Auto'].price,ax=ax[1])
ax[1].set_title('Price in Diesel Audi')
sns.distplot(audi[audi['transmission']=='Hybrid'].price,ax=ax[2])
ax[2].set_title('Price in Hybrib Audi')
plt.show()
sns.heatmap(audi.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


