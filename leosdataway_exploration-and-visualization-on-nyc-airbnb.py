# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
airbnb = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

airbnb.head(10)
airbnb.info()
airbnb.duplicated().sum()

airbnb.drop_duplicates(inplace=True)
airbnb.isnull().sum()
# replace nan reviews_per_month by zero

airbnb['reviews_per_month'].fillna(0,inplace=True)

airbnb['name'].fillna('noname',inplace=True)

# drop columns that are useless for analysis

airbnb.drop(['id','host_name','last_review'],axis=1,inplace=True)

airbnb.isnull().sum()
airbnb.dropna(how='any',inplace=True)

airbnb.info()
airbnb['neighbourhood'].value_counts()
airbnb['neighbourhood_group'].value_counts()
airbnb['room_type'].value_counts()
airbnb.describe(include='all')
top_host = airbnb['host_id'].value_counts().head(10)

top_host
plt.figure(figsize=(10,8))

sns.heatmap(airbnb.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)
sns.set()

# Top 10 room host

plt.figure(figsize=(10,6))

pic_host = sns.barplot(top_host.index,top_host.values,palette="Blues_d")

pic_host.set_title('Top 10 Host')

pic_host.set_xlabel('Host_id')

pic_host.set_ylabel('Amount of Airbnbs')
# Location of houses from the host who has most airbnb

most_host = airbnb[airbnb['host_id']==219517861]



nyc_img=mpimg.imread('/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png',0)

ax=plt.gca()

#scaling the image based on the latitude and longitude max and mins for proper output

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

most_host.plot(kind='scatter', x='longitude', y='latitude', label='neighbourhood_group',ax=ax, c='price',cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))
# Plot all Neighbourhood

# Which Neighbourhood group has the most airbnb?

sub_1=airbnb.loc[airbnb['neighbourhood_group'] == 'Brooklyn']

sub_2=airbnb.loc[airbnb['neighbourhood_group'] == 'Manhattan']

sub_3=airbnb.loc[airbnb['neighbourhood_group'] == 'Queens']

sub_4=airbnb.loc[airbnb['neighbourhood_group'] == 'Staten Island']

sub_5=airbnb.loc[airbnb['neighbourhood_group'] == 'Bronx']

plt.figure(figsize=(10,8))

pic_neigh = sns.barplot(airbnb['neighbourhood_group'].value_counts().index,airbnb['neighbourhood_group'].value_counts().values,palette="Blues_d")

pic_neigh.set_title('Amount of rooms in different neighbourhood_group')

pic_neigh.set_xlabel('Neighbourhood Groups')

pic_neigh.set_ylabel('Amount of Airbnbs')
# All airbnb distribution sorted by neighbourhood group

plt.figure(figsize=(10,8))

nyc_img=mpimg.imread('/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png',0)

ax=plt.gca()



#scaling the image based on the latitude and longitude max and mins for proper output

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])



sns.scatterplot(airbnb.longitude,airbnb.latitude,hue=airbnb.neighbourhood_group,

                ax=ax,alpha=0.4,cmap=plt.get_cmap('jet'))
# Room type distribution

plt.figure(figsize=(6,6))

pic_roomtype = sns.countplot(airbnb['room_type'])

# sns.barplot(top_host.index,top_host.values,palette="Blues_d")

pic_roomtype.set_title('Room Type')

pic_roomtype.set_xlabel('Room Type')

pic_roomtype.set_ylabel('Amount of Airbnbs')
# Room type distribution

room1 = sub_1['room_type'].value_counts()

room2 = sub_2['room_type'].value_counts()

room3 = sub_3['room_type'].value_counts()

room4 = sub_4['room_type'].value_counts()

room5 = sub_5['room_type'].value_counts()

room = pd.DataFrame([room1,room2,room3,room4,room5],index=['Brooklyn','Manhattan','Queens','Staten Island','Bronx'])

room.head()
room.plot(kind='bar',stacked=True,figsize=(10,8))
# Room type distribution

plt.figure(figsize=(10,8))

nyc_img=mpimg.imread('/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png',0)

ax=plt.gca()



#scaling the image based on the latitude and longitude max and mins for proper output

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])



sns.scatterplot(airbnb.longitude,airbnb.latitude,hue=airbnb.room_type,

                ax=ax,alpha=0.4,cmap=plt.get_cmap('jet'))
# Price in different Neighbourhood?



price_sub1=sub_1['price'] #Brooklyn

price_sub2=sub_2['price'] #Manhattan

price_sub3=sub_3['price'] #Queens

price_sub4=sub_4['price'] #Staten Island

price_sub5=sub_5['price'] #Bronx



plt.figure(figsize=(20,8))

sns.kdeplot(price_sub1[price_sub1<500],shade=True,label='Brooklyn')

sns.kdeplot(price_sub2[price_sub2<500],shade=True,label='Manhattan')

sns.kdeplot(price_sub3[price_sub3<500],shade=True,label='Queens')

sns.kdeplot(price_sub4[price_sub4<500],shade=True,label='Staten Island')

sns.kdeplot(price_sub5[price_sub5<500],shade=True,label='Bronx')

plt.title('Distribution of prices for each neighberhood_group',size=20)

plt.legend(prop={'size':16})
# Map of price

nyc_img=mpimg.imread('/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png',0)

ax=plt.gca()

#scaling the image based on the latitude and longitude max and mins for proper output

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

# drop price over 500

airbnb_price = airbnb[airbnb['price']<500] 

airbnb_price.plot(kind='scatter', x='longitude', y='latitude', label='neighbourhood_group',ax=ax, c='price',cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))
plt.figure(figsize=(10,15))

sns.boxplot(x='neighbourhood_group',y='number_of_reviews',data=airbnb)
# Map of views

nyc_img=mpimg.imread('/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png',0)

ax=plt.gca()

#scaling the image based on the latitude and longitude max and mins for proper output

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

# drop views over 120

airbnb_views = airbnb[airbnb['number_of_reviews']<120] 

airbnb_views.plot(kind='scatter', x='longitude', y='latitude',ax=ax, c='number_of_reviews',cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))
airbnb['availability_365'].median()
plt.figure(figsize=(10,8))

sns.boxplot(x='neighbourhood_group',y='availability_365',data=airbnb)
# Map of views

nyc_img=mpimg.imread('/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png',0)

ax=plt.gca()

#scaling the image based on the latitude and longitude max and mins for proper output

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])



airbnb.plot(kind='scatter', x='longitude', y='latitude',ax=ax, c='availability_365',cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))
# Word cloud of room name

from wordcloud import WordCloud

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(background_color='white',width=1920,height=1080).generate(" ".join(airbnb.name.str.upper()))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('roomname.png')

plt.show()
# Word cloud of top 50 views rooms' name

airbnb_top_view = airbnb.sort_values('number_of_reviews',ascending=False).iloc[:100]

from wordcloud import WordCloud

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(background_color='white',width=1920,height=1080).generate(" ".join(airbnb_top_view.name.str.upper()))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('roomname_topviews.png')

plt.show()
airbnb_ana = airbnb[['neighbourhood_group','room_type','price','number_of_reviews','availability_365']]

airbnb_ana.head(10)
# Mapping

airbnb_ana['neighbourhood_group'] = airbnb_ana['neighbourhood_group'].map({'Brooklyn':1,'Manhattan':2,'Queens':3,'Staten Island':4,'Bronx':5}).astype(int)

airbnb_ana['room_type'] = airbnb_ana['room_type'].map({'Private room':1,'Entire home/apt':2,'Shared room':3}).astype(int)

airbnb_ana.head(10)
plt.figure(figsize=(6,6))

sns.heatmap(airbnb_ana.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)
def Dropabnormal(df,columns,alpha):

    # use quantile to drop abnormal

    q_low = df[columns].quantile(q=0.25)

    q_high = df[columns].quantile(q=0.75)

    q_interval = q_high - q_low

    df = df[df[columns]<q_high+alpha*q_interval][df[columns]>q_low-alpha*q_interval]

    return df
# drop abnormal price and 

airbnb_ana = Dropabnormal(airbnb_ana,'price',3)

airbnb_ana = Dropabnormal(airbnb_ana,'number_of_reviews',3)

airbnb_ana.head(10)
airbnb_ana2 = airbnb_ana

plt.figure(figsize=(6,6))

sns.heatmap(airbnb_ana2.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,r2_score,mean_squared_error
x = np.array(airbnb_ana[['neighbourhood_group','room_type','number_of_reviews','availability_365']])

y = np.array(airbnb_ana['price'])

#Getting Test and Training Set

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=231)
# LinearRegression Model

from sklearn.linear_model import LinearRegression

lg = LinearRegression()

lg.fit(x_train,y_train)

y_pred = lg.predict(x_test)

print('LR:',r2_score(y_test,y_pred))

print('LR:',mean_squared_error(y_test,y_pred))

sns.regplot(y_pred,y_test)
# Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=346)

dt = DecisionTreeRegressor(min_samples_leaf=.0001)

dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)

print('DT:',r2_score(y_test,y_pred))

print('DT:',mean_squared_error(y_test,y_pred))

sns.regplot(y_pred,y_test)
# Lasso

from sklearn.linear_model import Lasso

la=Lasso(alpha=0.01)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1)    

la.fit(x_train,y_train)

print('Lasso:',r2_score(y_test,y_pred))

print('Lasso:',mean_squared_error(y_test,y_pred))

sns.regplot(y_pred,y_test)
# LogisticRegression

from sklearn.linear_model import LogisticRegression



lg = LogisticRegression()

lg.fit(x_train,y_train)

y_pred = lg.predict(x_test)

print('LR:',r2_score(y_test,y_pred))

print('LR:',mean_squared_error(y_test,y_pred))

sns.regplot(y_pred,y_test)