# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/fertilizers-by-product-fao/FertilizersProduct.csv', encoding='latin-1')

data.head()
data.describe()
data.isna().sum()
indonesia=data.drop(['Area Code','Item Code', 'Element Code', 'Year Code', 'Flag'],axis=1)

indonesia=indonesia.loc[indonesia.Area=='Indonesia']
indonesia
plt.figure(figsize=(20,8))

total = float(len(indonesia["Area"]))



ax = sns.countplot(x="Year", data=indonesia)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
indonesia.Element.dropna(inplace=True)

labels=indonesia.Element.value_counts().index

colors=['grey','blue','red','yellow','green','pink']

explode=[0,0,0,0,0,0]

sizes=indonesia.Element.value_counts().values



plt.figure(figsize=(10,10))

plt.pie(sizes,explode=explode,colors=colors,labels=labels,autopct='%1.1f%%')

plt.title('export or import elements in Indonesia',color='blue',fontsize=15)
plt.figure(figsize=(10,20))

total = float(len(indonesia))



ax = sns.countplot(y="Item", data=indonesia)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*120),

            ha="center") 

plt.show()
#rata-rata bunuh diri berdasarkan dekade dan gender

plt.figure(figsize=(20,5))

sns.barplot(x='Year',y='Value', hue='Unit',data=indonesia)

plt.xlabel('Year')

plt.ylabel('Value')

plt.title('units based on value and year')

plt.show()