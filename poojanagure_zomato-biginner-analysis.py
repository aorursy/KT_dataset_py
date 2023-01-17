# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/zomato.csv")

data.head()
data.info()
print('Number of rows = ',data.shape[0])

print('Number of columns = ', data.shape[1])
print('How many null values there?\n', data.isnull().sum())
pd.DataFrame({'Column':[i.upper() for i in data.columns],

             'Count':data.isnull().sum().values,

             'Percentage':((data.isnull().sum().values/len(data))*100).round(2)})
print('Dropping all the null values')

data.dropna(how='any',inplace=True)
data.isnull().sum()
print('some colums are uneccesary ')

data.drop(['url','menu_item','phone'],axis=1,inplace=True)
data['rate'] = data['rate'].replace('NEW',np.NaN)

data['rate'] = data['rate'].replace('-',np.NaN)

data['rate'] = data.loc[:,'rate'].replace('[ ]','',regex = True)

data['rate'] = data['rate'].astype(str)

data['rate'] = data['rate'].apply(lambda r: r.replace('/5',''))

data['rate'] = data['rate'].apply(lambda r: float(r))
#changing number line 1,200 to 1200for computing

data['approx_cost(for two people)']=data['approx_cost(for two people)'].str.replace(',','')

data['approx_cost(for two people)']=data['approx_cost(for two people)'].astype(int)
#Removing white spaces

data['name'] = data['name'].str.strip()

data['listed_in(city'] = data['listed_in(city)'].str.strip()

data['cuisines'] = data['cuisines'].str.strip()

data['listed_in(type)'] = data['listed_in(type)'].str.strip()
print('Restaurents delivering online order or not')

sns.countplot(x=data['online_order'])

fig=plt.gcf()

fig.set_size_inches(6,4)
plt.title('Restaurents providing table bookinf facility')

sns.countplot(x=data['book_table'])

fig=plt.gcf()

fig.set_size_inches(8,6)
sns.countplot(x=data['book_table'],hue= data['listed_in(type)'])

fig=plt.gcf()

fig.set_size_inches(10,10)

plt.title('Type of restusrents providing table booking facility')
sns.countplot(x=data['listed_in(type)'])

plt.title('Restaurent type', fontsize=15, fontweight='bold')

plt.xlabel('Restaurent Type',fontsize=10,fontweight='bold')

plt.xticks(rotation=45, fontsize=10,fontweight='bold')

plt.ylabel('')
plt.title('Most popular cuisines in Bangluru')

cuisines=data['cuisines'].value_counts()[:10]

sns.barplot(cuisines, cuisines.index)

plt.xlabel('Count')

fig.set_size_inches(15,10)
plt.figure(figsize=(12,10))

ax = sns.countplot(x='rate',hue='book_table',data=data)



plt.title('Table Booking - Ratings', fontsize = 15, fontweight='bold')

plt.xticks(fontsize=10,fontweight='bold')

plt.xlabel('Rating', fontsize=10,fontweight='bold')

plt.ylabel('Number of table bookings', fontsize=10, fontweight='bold')

plt.figure(figsize=(10,10))

ax=sns.countplot(x='rate',hue='online_order',data=data)

plt.title("Online order vs Rating")
print("no. of restaurants between 3.5 and 4 rating:")

((data.rate>=3.5) & (data.rate<4)).sum()
slices=[((data.rate>=1) & (data.rate<2)).sum(),

       ((data.rate>=2) & (data.rate<3)).sum(),

       ((data.rate>=3) & (data.rate<4)).sum(),

       ((data.rate>=4) & (data.rate<5)).sum()]

labels=['1-2','2-3','3-4','4-5']

colors = ['#3333cc','#ffff2a','#ff3333','#6699ff']

plt.pie(slices,colors=colors,labels=labels, autopct = '%1.0f%%', pctdistance=.5, labeldistance=1.2, shadow=True)

fig=plt.gcf()

plt.title('Percentage of restaurents according to ratings')

fig.set_size_inches(10,10)
print('Count of restaurents at unique locations')

locationCount=data['location'].value_counts().sort_values(ascending=True)

locationCount
print('Maximum no. of restauretns present at:')

count_max=max(locationCount)

for x,y in locationCount.items():

    if (y == count_max):

        print(x)
print('Minimum no. of reataurents present at:')

count_min=min(locationCount)

for x,y in locationCount.items():

    if(y== count_min):

        print(x)
fig=plt.figure(figsize=(20,40))

locationCount.plot(kind='barh',fontsize=20)

plt.ylabel('Location Names', fontsize=40, fontweight='bold', color='black')

plt.title('Restaurent Count graoh Vs Location', fontsize=40, color='black', fontweight='bold')

for i in range(len(locationCount)):

    plt.text(i+locationCount[i],i,locationCount[i],fontsize=10,fontweight='bold')
CityCount=data['listed_in(city)'].value_counts().sort_values(ascending=True)

fig=plt.figure(figsize=(20,20))

CityCount.plot(kind="barh",fontsize=20)

plt.ylabel("Location names",fontsize=50,color="red",fontweight='bold')

plt.title("CITY VS RESTAURANT COUNT GRAPH",fontsize=40,color="BLACK",fontweight='bold')

for i in range(len(CityCount)):

    

    plt.text(i+CityCount[i],i,CityCount[i],fontsize=10,color="BLACK",fontweight='bold')
#please suggest if any, to improve my kernel.