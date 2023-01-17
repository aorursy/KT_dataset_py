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
import json

#reads data into dictionary

data = [json.loads(line) for line in open('/kaggle/input/yelp-dataset/business.json', 'r')]
#Convert dictionary into a pandas df

import pandas as pd

df=pd.DataFrame.from_dict(data)

df.head(2)
##Explore the data in State

df['state'].unique()
import numpy as np

#First, isolate Nevada rows 

NV=df[(df['state']=='NV')]

#Then eliminate closed businesses from the df

NV=NV[(NV['is_open']==1)] 

#drop rows with 'None' data

NV = NV.replace(to_replace='None', value=np.nan).dropna()

NV.head(2)
#clean df

NV=NV.drop(['business_id', 'is_open'], axis=1)

NV.head(2)
%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
fig = plt.figure(figsize=(8, 8))

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=45, lon_0=-100,)

m.etopo(scale=0.5, alpha=0.5)



# Map (long, lat) to (x, y) for plotting

x, y = m(-117.224121, 39.876019)

plt.plot(x, y, 'ok', markersize=5)

plt.text(x, y, ' Nevada', fontsize=12)
#map of Greater LA with business markers

import folium

m = folium.Map(location=[36.1699, -115.1398], zoom_start=8)



NV.apply(lambda row:folium.CircleMarker(location=[row['latitude'], row['longitude']], 

                                              radius=2, popup=row['name'])

                                             .add_to(m), axis=1)



m
#focus on businesses with 3 stars and above

LV=NV[(NV['stars']>=3) & (NV['city']=='Las Vegas')] 

LV.head(2)
#Just those businesses with 20 or more reviews

LV=LV[(LV['review_count']>20)]
LV
#check the number of unique labels in the category column

LV['categories'].nunique()
#map the new data frame



n = folium.Map(location=[36.1699, -115.1398], zoom_start=10)



LV.apply(lambda row:folium.CircleMarker(location=[row['latitude'], row['longitude']], 

                                              radius=2, popup=row['name'])

                                             .add_to(n), axis=1)

n
#statistical description

LV.describe()
#turn columns to list

lat=LV['latitude']

lon=LV['longitude']



def merge(lat, lon): 

      #merge two lists into touple

    merged_list = tuple(zip(lat, lon))  

    return merged_list 



coord=list(merge(lat, lon)) 
def distance(instance1, instance2):

    # just in case, if the instances are lists or tuples:

    instance1 = np.array(instance1) 

    instance2 = np.array(instance2)

    

    return np.linalg.norm(instance1 - instance2)

dist=[]

for row in coord: 

    airport= (36.0840, -115.1537)

    d=(distance(row, airport))

    dist.append(d)

    print(d)

LV['Distance from airport']=dist

#need the data to be numerical

neighbor=LV[['name','stars','categories','review_count','Distance from airport','postal_code', 'latitude', 'longitude']]

neighbor
neighbor.info()
neighbor['postal_code']=neighbor['postal_code'].astype('category')

neighbor['zip code']=neighbor['postal_code'].cat.codes



neighbor['name']=neighbor['name'].astype('category')

neighbor['business']=neighbor['name'].cat.codes



neighbor['categories']=neighbor['categories'].astype('category')

neighbor['business type']=neighbor['categories'].cat.codes



neighbor.head()
Nums=neighbor.drop(['name','categories', 'postal_code'], axis=1)

Nums.head(3)
Nums.describe()
#visualize relative to 'stars' rating

import seaborn as sns



sns.pairplot(Nums, hue='stars') #colors defined by cats in single column
#split the dataset

from sklearn.model_selection import train_test_split



X = Nums.iloc[:,2-9].values.reshape(-1, 1)

y = Nums.iloc[:, 1].values #target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
#standardize data with scaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
### define number of neighbors and train the model



from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 3)

classifier.fit(X_train, y_train)
#predict

y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score



result1 = accuracy_score(y_test,y_pred)

print("Accuracy:",result1)  ##accuracy of knn model



result2 = classification_report(y_test, y_pred)

print("Classification Report:",)

print (result2)

##map categories with different color markers
#folium   

b1 = folium.Map(location = [36.1699, -115.1398], 

                                        zoom_start = 11) #Las Vegas

  

folium.Marker([36.0840, -115.1537], 

              popup = 'Airport').add_to(b1) #The airport

  

folium.Marker([36.1671, -115.1356], 

              popup = 'Downtown Las Vegas').add_to(b1) #downtown

  

# Add a line to the map   

folium.PolyLine(locations = [(36.0840, -115.1537), (36.1671, -115.1356)], 

                line_opacity = 0.5).add_to(b1) 

  

b1.save("map1.html") 



b1
#visualize the top businesses near the airport

near=neighbor[(neighbor['Distance from airport']<0.05)&(neighbor['stars']>4.5)]
near
b1



near.apply(lambda row:folium.CircleMarker(location=[row['latitude'], row['longitude']], 

                                              radius=2, popup=row['name'], zoom_start = 10)

                                             .add_to(b1), axis=1)



b1
folium.Marker(

    location=[36.082059, -115.172787],

    popup='Las Vegas SIGN',

    icon=folium.Icon(color='gray') 

).add_to(b1)

folium.Marker(

    location=[36.108725, -115.165826],

    popup='Nearest Hospital',

    icon=folium.Icon(color='red') 

).add_to(b1)

folium.Marker(

    location=[36.147247, -115.156031],

    popup='Stratosphere Hotel',

    icon=folium.Icon(color='purple') 

).add_to(b1)

b1