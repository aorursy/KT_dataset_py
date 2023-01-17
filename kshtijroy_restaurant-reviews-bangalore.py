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
data= pd.read_csv('/kaggle/input/zomato-bangalore-restaurants/zomato.csv')

import matplotlib.pyplot as plt

import seaborn as sns
data.head()
data.info()
data.columns
data.describe()
data.isnull().sum()
data.shape
data=data[data.cuisines.isna()==False]
data.shape
data.isnull().sum()
data.drop(columns=["url","address","phone","listed_in(city)"],inplace=True)
data.rename(columns={'approx_cost(for two people)': 'average_cost'}, inplace=True)
data.name.value_counts().head()
plt.figure(figsize=(12,6))

ax= data.name.value_counts()[:20].plot(kind='bar')

ax.legend(['*Restaurants'])

plt.xlabel("Name of Restaurant")

plt.ylabel("Count of Restaurants")

plt.title("Name vs Number of Restaurant",fontsize =20, weight = 'bold')
data.online_order.value_counts()
ax= sns.countplot(data['online_order'])

plt.title('Number of Restaurants accepting online orders', weight='bold')

plt.xlabel('online orders')
data.book_table.value_counts()
sns.countplot(data['book_table'])

plt.title("No of Restaurant with Book Table Facility", weight = 'bold')

plt.xlabel('Book table facility')

plt.ylabel('No of restaurants')
data.location.value_counts()
plt.figure(figsize=(12,10))

data['location'].value_counts()[:10].plot(kind='pie')

plt.title('Location', weight = 'bold')
plt.figure(figsize=(12,10))

data.location.value_counts()[:10].plot(kind='bar')

plt.title("Location vs Count", weight = 'bold')
data.location.nunique()
data['rest_type'].value_counts().head()
plt.figure(figsize=(14,8))

data.rest_type.value_counts()[:10].plot(kind='pie')

plt.title('Restaurent Type', weight = 'bold')

plt.show()
data.average_cost.value_counts().head(20)
plt.figure(figsize=(12,8))

data['average_cost'].value_counts()[:20].plot(kind='pie',autopct='%1.1f%%')

plt.title('Avg cost in Restaurent for 2 people', weight = 'bold')
dishes_data = data[data.dish_liked.notnull()]

dishes_data.dish_liked = dishes_data.dish_liked.apply(lambda x:x.lower().strip())
dishes_data.isnull().sum()

dish_count = []

for i in dishes_data.dish_liked:

    for t in i.split(','):

        t = t.strip() # remove the white spaces to get accurate results

        dish_count.append(t)
plt.figure(figsize=(12,6)) 

pd.Series(dish_count).value_counts()[:10].plot(kind='bar',color= 'c')

plt.title('Top 10 dished_liked in Bangalore',weight='bold')

plt.xlabel('Dish')

plt.ylabel('Count')

cuisines_data = data[data.cuisines.notnull()]

cuisines_data.cuisines = cuisines_data.cuisines.apply(lambda x:x.lower().strip())
cuisines_count= []



for i in cuisines_data.cuisines:

    for j in i.split(','):

        j = j.strip()

        cuisines_count.append(j)

plt.figure(figsize=(12,6)) 

pd.Series(cuisines_count).value_counts()[:10].plot(kind='bar',color= 'r')

plt.title('Top 10 cuisines in Bangalore',weight='bold')

plt.xlabel('cuisines type')

plt.ylabel('No of restaurants')
data['rate'] = data['rate'].replace('NEW',np.NaN)

data['rate'] = data['rate'].replace('-',np.NaN)

data.dropna(how = 'any', inplace = True)
data['rate'] = data.loc[:,'rate'].replace('[ ]','',regex = True)

data['rate'] = data['rate'].astype(str)

data['rate'] = data['rate'].apply(lambda r: r.replace('/5',''))

data['rate'] = data['rate'].apply(lambda r: float(r))
data.rate.hist(color='grey')

plt.axvline(x= data.rate.mean(),ls='--',color='yellow')

plt.title('Average Rating for Bangalore Restaurants',weight='bold')

plt.xlabel('Rating')

plt.ylabel('No of Restaurants')

print(data.rate.mean())
data['online_order']= pd.get_dummies(data.online_order, drop_first=True)

data['book_table']= pd.get_dummies(data.book_table, drop_first=True)

data
data.drop(columns=['dish_liked','reviews_list','menu_item','listed_in(type)'], inplace  =True)
data['rest_type'] = data['rest_type'].str.replace(',' , '') 

data['rest_type'] = data['rest_type'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))

data['rest_type'].value_counts().head()

data['cuisines'] = data['cuisines'].str.replace(',' , '') 

data['cuisines'] = data['cuisines'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))

data['cuisines'].value_counts().head()
from sklearn.preprocessing import LabelEncoder

T = LabelEncoder()                 

data['location'] = T.fit_transform(data['location'])

data['rest_type'] = T.fit_transform(data['rest_type'])

data['cuisines'] = T.fit_transform(data['cuisines'])
data["average_cost"] = data["average_cost"].str.replace(',' , '') 
data["average_cost"] = data["average_cost"].astype('float')
x = data.drop(['rate','name'],axis = 1)
y = data['rate']

x.shape
y.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 33)
from sklearn.preprocessing import StandardScaler

num_values1=data.select_dtypes(['float64','int64']).columns

scaler = StandardScaler()

scaler.fit(data[num_values1])

data[num_values1]=scaler.transform(data[num_values1])
data.head()
from sklearn import metrics

from sklearn.tree import DecisionTreeRegressor 

dc =  DecisionTreeRegressor()

dc.fit(X_train,y_train)

y_pred_rfr = dc.predict(X_test)
dc.score(X_test,y_test)*100
