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
# importing important libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn

plt.style.use('ggplot')



%matplotlib inline
# reading the data

data_abb=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data_abb.head()
data_abb.reset_index()
data_abb.drop('id',axis=1,inplace=True)
data_abb.head()
data_abb.columns
data_abb=data_abb[['name', 'host_id', 'host_name', 'neighbourhood_group', 'neighbourhood',

       'latitude', 'longitude', 'room_type', 'minimum_nights',

       'number_of_reviews', 'last_review', 'reviews_per_month',

       'calculated_host_listings_count', 'availability_365','price']]
data_abb.head()
data_abb.shape
data_abb.host_id.nunique()
#checking the data

data_abb.info()
data_abb.isna().sum()
data_abb.last_review
# converting last_review to datetime

# replacing NaN values with 0 

data_abb['last_review']=pd.to_datetime(data_abb['last_review'])

data_abb.last_review.fillna(max(data_abb.last_review),inplace=True)

data_abb.reviews_per_month.fillna(0,inplace=True)
#removing unwanted columns

data_abb.drop(['name','host_name'],axis=1,inplace=True)
#checking if any null values present now

data_abb.isna().sum().sum()
data_abb.describe()
data_abb.info()
data_abb.head()
data_abb.host_id.nunique()
#let's see what hosts (IDs) have the most listings on Airbnb platform and taking advantage of this service

top_host=data_abb.host_id.value_counts().head(10)

top_host
#coming back to our dataset we can confirm our fidnings with already existing column called 'calculated_host_listings_count'

top_host_check=data_abb.calculated_host_listings_count.max()

top_host_check
#setting figure size for future visualizations

sns.set(rc={'figure.figsize':(10,8)})

viz_1=top_host.plot(kind='bar',cmap='plasma')

viz_1.set_title('Hosts with the most listings in NYC')

viz_1.set_ylabel('Count of listings')

viz_1.set_xlabel('Host IDs')

viz_1.set_xticklabels(viz_1.get_xticklabels(), rotation=45)
a=data_abb.groupby('neighbourhood_group').calculated_host_listings_count.sum()


plt.style.use('ggplot')

a.plot(kind='bar')
sns.countplot(x='neighbourhood_group',data=data_abb)
data_abb.groupby('neighbourhood_group')['neighbourhood'].nunique().plot(kind='bar',colormap='Set3')

plt.xlabel('neighbourhood groups')

plt.ylabel('neighbourhoods')

print(data_abb.groupby('neighbourhood_group')['neighbourhood'].nunique())
print('Total neighbourhoods in NYC in which listings are located: {}'.format(data_abb.neighbourhood.value_counts().sum()))



plt.figure(figsize=(24,12))

# Top 10 neighbourhoods in NYC

plt.subplot(2,1,1)

V2=sns.countplot(y='neighbourhood',                                            #Create a Horizontal Plot

                   data=data_abb,                                                    

                   order=data_abb.neighbourhood.value_counts().iloc[:10].index,      #We want to view the top 10 Neighbourhoods

                   edgecolor=(0,0,0),                                            #This cutomization gives us black borders around our plot bars

                   linewidth=2)

V2.set_title('Listings by Top NYC Neighbourhood')                                #Set Title

V2.set_xlabel('Neighbourhood')                                  

V2.set_ylabel('Listings')



# 10 Least preferred neighbourhood in NYC

plt.subplot(2,1,2)

V3 = sns.countplot(y='neighbourhood',                                            #Create a Horizontal Plot

                   data=data_abb,                                                    

                   order=data_abb.neighbourhood.value_counts().iloc[-10:].index,      #We want to view the top 10 Neighbourhoods

                   edgecolor=(0,0,0),                                            #This cutomization gives us black borders around our plot bars

                   linewidth=2)

V3.set_title('Listings by Least Preffered NYC Neighbourhood')                                #Set Title

V3.set_xlabel('Neighbourhood')                                  

V3.set_ylabel('Listings')
data_abb.room_type.unique()
sns.countplot(x='room_type',data=data_abb,edgecolor=sns.color_palette("dark", 3))

data_abb.room_type.value_counts()
print('Percentage of room types available in AirBnB registered listings are:\n {}'.format((data_abb.room_type.value_counts()/len(data_abb.room_type))*100))
b=data_abb.room_type.value_counts()/len(data_abb.room_type)

b.plot.pie(autopct='%.2f',fontsize=12,figsize=(8,8))

plt.title('Room types availability in AirBnB',fontsize=20)
data_abb.groupby(['neighbourhood_group','room_type']).room_type.count().plot.barh(stacked=True)

plt.ylabel('Neighbourhood wise room types')

plt.xlabel('Number of Rooms')

plt.title('Neighbourhood groups Vs Room types availability')
#let's now combine this with our boroughs and room type for a rich visualization we can make



#grabbing top 10 neighbourhoods for sub-dataframe

sub=data_abb.loc[data_abb['neighbourhood'].isin(['Williamsburg','Bedford-Stuyvesant','Harlem','Bushwick',

                 'Upper West Side','Hell\'s Kitchen','East Village','Upper East Side','Crown Heights','Midtown'])]

#using catplot to represent multiple interesting attributes together and a count

viz=sns.catplot(x='neighbourhood', hue='neighbourhood_group', col='room_type', data=sub, kind='count')

viz.set_xticklabels(rotation=90)
data_abb.groupby('neighbourhood_group').price.describe()
data_abb
# setting up bins for price in order to have better understanding of rooms distribution

data_abb['price_range']=pd.qcut(data_abb['price'],10)
list(data_abb.neighbourhood_group.unique())
neighbourhood_group=list(data_abb.neighbourhood_group.unique())

plt.figure(figsize=(40,36))

for i,neighbour in enumerate(neighbourhood_group):

    plt.subplot(3,2,i+1)

    sns.countplot(y='price_range',hue='room_type',data=data_abb[data_abb['neighbourhood_group']==neighbour])

    plt.xlabel('Number of listings')

    plt.ylabel('Price range in which listings fall')

    plt.title('Price listings vs neighbourhood and room types in {}'.format(neighbour))
g = data_abb[data_abb.price <500]

plt.figure(figsize=(10,6))

sns.boxplot(y="price",x ='neighbourhood_group' ,data = g)

plt.title("neighbourhood_group price distribution < 500")

plt.show()
plt.figure(figsize=(10,6))

sns.distplot(data_abb[data_abb.neighbourhood_group=='Manhattan'].price,color='maroon',hist=False,label='Manhattan')

sns.distplot(data_abb[data_abb.neighbourhood_group=='Brooklyn'].price,color='black',hist=False,label='Brooklyn')

sns.distplot(data_abb[data_abb.neighbourhood_group=='Queens'].price,color='green',hist=False,label='Queens')

sns.distplot(data_abb[data_abb.neighbourhood_group=='Staten Island'].price,color='blue',hist=False,label='Staten Island')

sns.distplot(data_abb[data_abb.neighbourhood_group=='Long Island'].price,color='lavender',hist=False,label='Long Island')

plt.title('Borough wise price destribution for price<2000')

plt.xlim(0,2000)

plt.show()
rooms=list(data_abb.room_type.unique())

for i,room in enumerate(rooms):

    plt.figure(figsize=(60,8))

    plt.subplot(1,3,i+1)

    sns.barplot(y='price_range',x='minimum_nights',data=data_abb[(data_abb.room_type==room)])

    

    plt.title(room)
sns.scatterplot(x='minimum_nights',y='price',data=data_abb)
data_abb.groupby(['room_type','price_range'])['minimum_nights'].describe()
plt.figure(figsize=(10,6))

sns.scatterplot(data_abb.longitude,data_abb.latitude,hue=data_abb.neighbourhood_group)

plt.ioff()
import folium

from folium.plugins import HeatMap

m=folium.Map([40.7128,-74.0060],zoom_start=11)

HeatMap(data_abb[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)
#let's grab 100 most reviewed listings in NYC

top_reviewed_listings=data_abb.nlargest(100,'number_of_reviews')

top_reviewed_listings
price_avrg=top_reviewed_listings.price.mean()

print('Average price per night: {}'.format(price_avrg))
top_reviewed_listings.groupby('room_type')['price'].describe()


sns.boxplot(y='price',x='room_type',data=top_reviewed_listings)
sns.distplot(data_abb[(data_abb['minimum_nights'] <= 30) & (data_abb['minimum_nights'] > 0)]['minimum_nights'], bins=31)

plt.ioff()
plt.figure(figsize=(10,6))

plt.scatter(data_abb.longitude, data_abb.latitude, c=data_abb.availability_365, cmap='spring', edgecolor='black', linewidth=1, alpha=0.75)



cbar = plt.colorbar()

cbar.set_label('availability_365')
data_abb.corr()
# We will make model to only use listings which has price set up. Their are multiple listings with no prices. 

# We will also use listings which has availability_365>0

data_abb=data_abb[data_abb.price>0]

data_abb=data_abb[data_abb.availability_365>0]
# Setting the target variable and independent variable

X=['latitude','longitude','minimum_nights','number_of_reviews','availability_365','room_type','neighbourhood_group','neighbourhood']

y='price'
data_X=data_abb[X]

data_X.head()
data_y=data_abb[y]
data_X.head()
# encoding the categorical data for making data suitable for machine to learn

X=pd.get_dummies(data_X,prefix_sep='_',drop_first=True)
X.shape
y=np.log10(data_y)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)
# importing important LinearRegression ML models

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)



y_pred=lr.predict(X_test)
# Evaluation of model



from sklearn.metrics import mean_squared_error

from sklearn import metrics

from sklearn.metrics import r2_score



print('RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, lr.predict(X_test))), 2))

print('R2 score train:', np.round(r2_score(y_train, lr.predict(X_train), multioutput='variance_weighted'), 2))

print('R2 score test:', np.round(r2_score(y_test, lr.predict(X_test), multioutput='variance_weighted'), 2))
sns.distplot(X.minimum_nights)
from scipy.stats import norm

log_min_night=np.log(X.minimum_nights)



sns.distplot(log_min_night,fit=norm)
from sklearn.linear_model import BayesianRidge

br=BayesianRidge()

br.fit(X_train,y_train)

y_predict=br.predict(X_test)
print('RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, lr.predict(X_test))), 2))

print('R2 score train:', np.round(r2_score(y_train, lr.predict(X_train), multioutput='variance_weighted'), 2)*100)

print('R2 score test:', np.round(r2_score(y_test, lr.predict(X_test), multioutput='variance_weighted'), 2))
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()                                            # Fit label encoder

le.fit(data_abb['neighbourhood_group'])

data_abb['neighbourhood_group']=le.transform(data_abb['neighbourhood_group'])    # Transform labels to normalized encoding.



le = LabelEncoder()

le.fit(data_abb['neighbourhood'])

data_abb['neighbourhood']=le.transform(data_abb['neighbourhood'])



le =LabelEncoder()

le.fit(data_abb['room_type'])

data_abb['room_type']=le.transform(data_abb['room_type'])



data_abb.sort_values(by='price',ascending=True,inplace=True)



data_abb.head()
lm = LinearRegression()



X = data_abb[['neighbourhood_group','neighbourhood','room_type','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']]

y = np.log10(data_abb['price'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)



lm.fit(X_train,y_train)
from sklearn.metrics import mean_absolute_error

y_predicts = lm.predict(X_test)



print("""

        Mean Squared Error: {}

        R2 Score: {}

        Mean Absolute Error: {}

     """.format(

        np.sqrt(metrics.mean_squared_error(y_test, y_predicts)),

        r2_score(y_test,y_predicts) * 100,

        mean_absolute_error(y_test,y_predicts)

        ))
from sklearn.tree import DecisionTreeRegressor



Reg_tree=DecisionTreeRegressor(criterion='mse',max_depth=3,random_state=0)

Reg_tree=Reg_tree.fit(X_train,y_train)



y_predicts=Reg_tree.predict(X_test)

print("median absolute deviation (MAD): ",np.mean(abs(np.multiply(np.array(y_test.T-y_predicts),np.array(1/y_test)))))

print("""

        Mean Squared Error: {}

        R2 Score: {}

        Mean Absolute Error: {}

     """.format(

        np.sqrt(metrics.mean_squared_error(y_test, y_predicts)),

        r2_score(y_test,y_predicts) * 100,

        mean_absolute_error(y_test,y_predicts)

        ))
from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

from IPython.display import Image as PImage

from sklearn.tree import export_graphviz

with open("tree1.dot", 'w') as f:

     f = export_graphviz(Reg_tree,

                              out_file=f,

                              max_depth = 3,

                              impurity = True,

                              feature_names = ['neighbourhood_group','neighbourhood','room_type','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365'],

                              rounded = True,

                              filled= True )

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")
y=np.log10(data_abb.price)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

Reg_tree.fit(X_train,y_train)

y_predicts=Reg_tree.predict(X_test)

from sklearn.metrics import r2_score

print('r2 score:',r2_score(y_test,y_predicts)*100,'%')

X = data_abb[['neighbourhood_group','neighbourhood','room_type','number_of_reviews','reviews_per_month','availability_365']]

y=np.log10(data_abb.price)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

Reg_tree.fit(X_train,y_train)

y_predicts=Reg_tree.predict(X_test)

from sklearn.metrics import r2_score

print('r2 score:',r2_score(y_test,y_predicts)*100,'%')
lm.fit(X_train,y_train)

y_predicts = lm.predict(X_test)



print("""

        Mean Squared Error: {}

        R2 Score: {}

        Mean Absolute Error: {}

     """.format(

        np.sqrt(metrics.mean_squared_error(y_test, y_predicts)),

        r2_score(y_test,y_predicts) * 100,

        mean_absolute_error(y_test,y_predicts)

        ))

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingRegressor

clf = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,

max_depth=1, random_state=0)

clf.fit(X_train,y_train)

scores = cross_val_score(clf, X_test, y_test, cv=5)

scores.mean()
y_predicts = clf.predict(X_test)



print("""

        Mean Squared Error: {}

        R2 Score: {}

        Mean Absolute Error: {}

     """.format(

        np.sqrt(metrics.mean_squared_error(y_test, y_predicts)),

        r2_score(y_test,y_predicts) * 100,

        mean_absolute_error(y_test,y_predicts)

        ))
X_test
y_test
y_test=10**y
y_test
X_test['Price']=y_test
X_test