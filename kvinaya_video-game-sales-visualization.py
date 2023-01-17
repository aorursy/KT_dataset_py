# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
vgs=pd.read_csv("../input/vgsales.csv")

vgs.head()
vgs=vgs.dropna(axis=0,how='any')

vgs.shape
vgs.info()
vgs['Genre'].unique()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.countplot(x='Genre',data=vgs)

plt.title('Video Game Genres count',fontsize=16)

plt.xticks(rotation=60)

plt.show()
sns.countplot(x='Platform',data=vgs)

plt.title('Video Game Platform count',fontsize=16)

plt.xticks(rotation=60)

plt.show()
from collections import Counter

genre = Counter(vgs['Genre']).most_common(10)

genre_name = [name[0] for name in genre]

genre_counts = [name[1] for name in genre]



fig,ax = plt.subplots(figsize=(8,5))

sns.barplot(x=genre_name,y=genre_counts,ax=ax)

plt.title('Top ten Genres',fontsize=18,fontweight='bold')

plt.xlabel('Genre',fontsize=15)

plt.ylabel('Number of counts',fontsize=15)

plt.xticks(rotation=60,fontsize=10)

plt.show()

from collections import Counter

genre = Counter(vgs['Platform']).most_common(10)

genre_name = [name[0] for name in genre]

genre_counts = [name[1] for name in genre]



fig,ax = plt.subplots(figsize=(8,5))

sns.barplot(x=genre_name,y=genre_counts,ax=ax)

plt.title('Top ten Platforms',fontsize=18,fontweight='bold')

plt.xlabel('Platform',fontsize=15)

plt.ylabel('Number of counts',fontsize=15)

plt.xticks(rotation=60,fontsize=10)

plt.show()

Year_list=vgs['Year'].unique()

sale = []



for Year in Year_list:

    sale.append(vgs[vgs['Year'] == Year]['Global_Sales'].sum())

    

    

fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x=Year_list,y=sale,ax=ax)

plt.xlabel('Year',fontsize=15)

plt.ylabel('total sales',fontsize=15)

plt.title('Global sales',fontsize=15)

plt.xticks(rotation=90)

plt.show()





Year_list=vgs['Year'].unique()

sale = []



for Year in Year_list:

    sale.append(vgs[vgs['Year'] == Year]['NA_Sales'].sum())

    

    

fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x=Year_list,y=sale,ax=ax)

plt.xlabel('Year',fontsize=15)

plt.ylabel('total sales',fontsize=15)

plt.title('NA sales by year',fontsize=15)

plt.xticks(rotation=90)

plt.show()



Year_list=vgs['Year'].unique()

sale = []



for Year in Year_list:

    sale.append(vgs[vgs['Year'] == Year]['EU_Sales'].sum())

    

    

fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x=Year_list,y=sale,ax=ax)

plt.xlabel(' Year',fontsize=15)

plt.ylabel('total sales',fontsize=15)

plt.title('EU sales by year',fontsize=15)

plt.xticks(rotation=90)

plt.show()



Year_list=vgs['Year'].unique()

sale = []



for Year in Year_list:

    sale.append(vgs[vgs['Year'] == Year]['JP_Sales'].sum())

    

    

fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x=Year_list,y=sale,ax=ax)

plt.xlabel(' Year',fontsize=15)

plt.ylabel('total sales',fontsize=15)

plt.title('JP sales by year',fontsize=15)

plt.xticks(rotation=90)

plt.show()
fig,ax = plt.subplots(figsize = (8,6))

vgs.groupby('Genre').sum()['NA_Sales'].plot(kind='bar',x='Genre',y="Sales",title='NA_Sales for different Genre')
fig,ax = plt.subplots(figsize = (8,6))

vgs.groupby('Genre').sum()["EU_Sales"].plot(kind="bar",x="Genre",y="Sales",title="EU_Sales for different Genre")

fig,ax = plt.subplots(figsize = (8,6))

vgs.groupby('Genre').sum()["JP_Sales"].plot(x="Genre", y="Sales", kind='bar', title="JP_Sales for different Genre")

fig,ax = plt.subplots(figsize = (8,6))

vgs.groupby('Genre').sum()['Global_Sales'].plot(x="Genre", y="Sales", kind='bar', title="Global_Sales for different Genre")

fig,ax = plt.subplots(figsize = (8,6))

vgs.groupby('Platform').sum()['Global_Sales'].plot(x="Platform", y="Sales", kind='bar', title="Global_Sales for different platform")

fig,ax=plt.subplots(nrows=3,ncols=3,figsize=(8,6))

fig.subplots_adjust(right=1.5,top=1,wspace = 0.5,hspace =1)

plt.subplot(3,3,1)

vgs[vgs['Platform']=='PS2']["Year"].value_counts().plot(kind='bar')

plt.title('Platform PS2',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)





plt.subplot(3,3,2)

vgs[vgs['Platform']=='PS']["Year"].value_counts().plot(kind='bar')

plt.title('Platform PS',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)





plt.subplot(3,3,3)

vgs[vgs['Platform']=='PC']["Year"].value_counts().plot(kind='bar')

plt.title('Platform PC',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)







plt.subplot(3,3,4)

vgs[vgs['Platform']=='DS']["Year"].value_counts().plot(kind='bar')

plt.title('Platform DS',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)





plt.subplot(3,3,5)

vgs[vgs['Platform']=='XB']["Year"].value_counts().plot(kind='bar')

plt.title('Platform XB',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)





plt.subplot(3,3,6)

vgs[vgs['Platform']=='Wii']["Year"].value_counts().plot(kind='bar')

plt.title('Platform Wii',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)





plt.subplot(3,3,7)

vgs[vgs['Platform']=='SNES']["Year"].value_counts().plot(kind='bar')

plt.title('Platform SNES',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)







plt.subplot(3,3,8)

vgs[vgs['Platform']=='SAT']["Year"].value_counts().plot(kind='bar')

plt.title('Platform SAT',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)





plt.subplot(3,3,9)

vgs[vgs['Platform']=='PSV']["Year"].value_counts().plot(kind='bar')

plt.title('Platform PSV',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)

plt.show()





fig,ax=plt.subplots(nrows=3,ncols=3,figsize=(10,6))

fig.subplots_adjust(left=0.2,right=1.2,top=1.2,wspace = 0.6,hspace =1.5)

plt.subplot(3,3,1)

vgs[vgs['Platform']=='PS2']["Genre"].value_counts().plot(kind='bar')

plt.title('platform PS2',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)





plt.subplot(3,3,2)

vgs[vgs['Platform']=='PS']["Genre"].value_counts().plot(kind='bar')

plt.title('platform PS',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)



plt.subplot(3,3,3)

vgs[vgs['Platform']=='PC']["Genre"].value_counts().plot(kind='bar')

plt.title('platform PC',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)



plt.subplot(3,3,4)

vgs[vgs['Platform']=='DS']["Genre"].value_counts().plot(kind='bar')

plt.title('platform DS',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)





plt.subplot(3,3,5)

vgs[vgs['Platform']=='XB']["Genre"].value_counts().plot(kind='bar')

plt.title('platform XB',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)





plt.subplot(3,3,6)

vgs[vgs['Platform']=='Wii']["Genre"].value_counts().plot(kind='bar')

plt.title('platform Wii',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)





plt.subplot(3,3,7)

vgs[vgs['Platform']=='GB']["Genre"].value_counts().plot(kind='bar')

plt.title('platform GB',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)





plt.subplot(3,3,8)

vgs[vgs['Platform']=='PSP']["Genre"].value_counts().plot(kind='bar')

plt.title('platform PSP',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)





plt.subplot(3,3,9)

vgs[vgs['Platform']=='SNES']["Genre"].value_counts().plot(kind='bar')

plt.title('platform SNES',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('count',fontsize=15)
