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
df=pd.read_csv('../input/student-alcohol-consumption/student-mat.csv')
df.head()
df.columns


df.drop(['address','reason','guardian'],axis=1,inplace=True)
df.info()
df.corr()
df.head(10)
data1=df.head(100)

# lineplot icin 100 sample sectik

data2=df.tail(100)
import matplotlib.pyplot as plt

data1.Medu.plot(kind='line',color='g',label='Mother-education',linewidth=1,grid=True)

data1.G1.plot(kind='line',color='r',label='Grade1',linewidth=1,grid=True)

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.legend()

plt.show()

data2.Medu.plot(kind='line',color='g',label='Mother-education',linewidth=1,grid=True)

data2.G1.plot(kind='line',color='r',label='Grade1',linewidth=1,grid=True)

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.legend()

plt.show()

import matplotlib.pyplot as plt

data1.Fedu.plot(kind='line',color='orange',label='Father-education',linewidth=1,grid=True)

data1.G1.plot(kind='line',color='blue',label='G1',linewidth=1,grid=True)

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.legend()

plt.show()



data2.Fedu.plot(kind='line',color='orange',label='Father-education',linewidth=1,grid=True)

data2.G1.plot(kind='line',color='blue',label='G1',linewidth=1,grid=True)

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.legend()

plt.show()
data2.Dalc.plot(kind='line',color='orange',label='Dalc',linewidth=1,grid=True)

data2.G1.plot(kind='line',color='blue',label='G1',linewidth=1,grid=True)

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.legend()

plt.show()

# günlül alkol tüketimi ile 1. not arasında ters ilişki mevcut
data2.plot(kind='scatter',x='studytime',y='G1',alpha=0.5,color='red')

plt.xlabel('Study-time')

plt.ylabel('G1')

plt.title(' Scatter Plot')

plt.show()

# çalışma saati ve g1 notu arasında çok yüksek bir korelasyon bulunmuyor.
data2.studytime.plot(kind='hist',bins=50,figsize=(7,7))

plt.show()

# öğrencilerin çoğu haftada 2 saat ders çalışıyor.
a=data1['G1'][data1['studytime']==2]

print(a.mean())

b=data1['G1'][data1['studytime']==4]

print(b.mean())

# haftada 2 saat ders çalışanlarla 4 saat çalışanlar arasinda sadece 1.5 puan fark var 
data2_demo=data2.head()

data2_demo_school=data2_demo.school

data2_demo_age=data2_demo.age

z=zip(data2_demo_school,data2_demo.age)

z_list=list(z)

print(z_list)


data2_demo['G1_basari']=[ 'dusuk' if i<data2_demo.G1.mean() else 'yuksek'  for i in data2_demo.G1 ]
print(df['famsize'].value_counts(dropna=False))

# family size analysis 

print(df['Pstatus'].value_counts(dropna=False))

# parent's cohabitation status 
print(df['Medu'].value_counts(dropna=False))

# Mother education 
print(df['Fedu'].value_counts(dropna=False))

# Father education 
df['G4(mean_all_grades)']=(df.G1+df.G2+df.G3)/3

df.head()

import matplotlib.pyplot as plt 

df.boxplot('G4(mean_all_grades)',by='sex')

plt.show()
data1_new=data1.head()

data1_new['name']=['a','b','c','d','e']

melted=pd.melt(frame=data1_new,id_vars='name',value_vars=['health','Dalc'])

melted
melted.pivot(index='name',columns='variable',values='value')
#vertical

data2_new=data2.head()

conc_data_row=pd.concat([data1_new,data2_new],axis=0,ignore_index=True)

conc_data_row
# horizontal

conc_data_col=pd.concat([data1_new.Dalc,data1_new.Walc],axis=1)

conc_data_col
# convert dtypes

data1_new['age'].dtypes

data1_new['age']=data1_new['age'].astype('float32')

data1_new['age'].dtypes