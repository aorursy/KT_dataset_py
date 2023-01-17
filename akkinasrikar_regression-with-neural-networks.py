#import the libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import style

style.use("fivethirtyeight")

import warnings

warnings.filterwarnings("ignore")

import folium

import webbrowser

from folium.plugins import HeatMap


#load the house data

housedata=pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")

housedata.head()
housedata.describe().transpose()
housedata.isnull().sum()
sns.heatmap(housedata.isnull(),yticklabels=False,cbar=False,cmap="viridis")

#we dont have any null values
housedata.corr()
sns.heatmap(housedata.corr(),cmap='rocket',cbar=True,yticklabels=True)
housedata.corr()["price"].sort_values()
plt.figure(figsize=(10,6))

sns.scatterplot(x="price",y="sqft_living",data=housedata,color="g",palette='viridis')
fig=plt.figure(figsize=(12,5))

axis=fig.add_subplot(121)

sns.distplot(housedata['price'],color='g')

plt.ylim(0,None)

plt.xlim(0,2000000)

axis.set_title('distribution of prices')



axis=fig.add_subplot(122)

sns.distplot(housedata['sqft_living'],color='b')

plt.ylim(0,None)

plt.xlim(0,6000)

axis.set_title('distribution of sqft_living')
plt.figure(figsize=(10,6))

sns.jointplot(x='sqft_living',y='price',kind='hex',data=housedata)

plt.ylim(0,3500000)

plt.xlim(0,None)
plt.figure(figsize=(10,6))

sns.lmplot(x='sqft_living',y='price',palette='viridis',height=7,data=housedata)

plt.title('sqft_living vs price')
plt.figure(figsize=(10,6))

sns.countplot(housedata["bedrooms"])
plt.figure(figsize=(10,6))

sns.countplot(housedata["grade"])
plt.figure(figsize=(10,6))

sns.boxplot(x='bedrooms',y='price',palette='viridis',data=housedata)

plt.title("bedrooms vs price")
fig=plt.figure(figsize=(19,12.5))

ax=fig.add_subplot(2,2,1,projection='3d')

ax.scatter(housedata['floors'],housedata['bedrooms'],housedata['bathrooms'],c="blue")

ax.set(xlabel='\nFloors',ylabel='\nBedrooms',zlabel='\nBathrooms')

ax.set(ylim=[0,12])



ax=fig.add_subplot(2,2,2,projection='3d')

ax.scatter(housedata['price'],housedata['sqft_living'],housedata['bedrooms'],c="green")

ax.set(xlabel='\nprice',ylabel='\nsqt_living',zlabel='\nBedrooms')

ax.set(zlim=[0,12])



ax=fig.add_subplot(2,2,3,projection='3d')

ax.scatter(housedata['floors'],housedata['bedrooms'],housedata['sqft_living'],c="red")

ax.set(xlabel='\nFloors',ylabel='\nBedrooms',zlabel='\nsqft_living')

ax.set(ylim=[0,12])



ax=fig.add_subplot(2,2,4,projection='3d')

ax.scatter(housedata['grade'],housedata['price'],housedata['sqft_living'],c="violet")

ax.set(xlabel='\ngrade',ylabel='\nprice',zlabel='\nsqft_living')

ax.set(xlim=[2,12])
plt.figure(figsize=(10,6))

sns.scatterplot(x="price",y="long",data=housedata,color="r")
plt.figure(figsize=(10,6))

sns.scatterplot(x="price",y="lat",data=housedata)
plt.figure(figsize=(10,6))

sns.scatterplot(x="long",y="lat",data=housedata,hue="price")
new_data=housedata.sort_values("price",ascending=False).iloc[200:]

#we are removing outliers
plt.figure(figsize=(10,6))

sns.scatterplot(x="long",y="lat",data=new_data,hue="price",palette="RdYlGn",alpha=0.2,edgecolor=None)
latitude=47.6

longitude=-122.3

dup=housedata.copy()

def worldmap(location=[latitude,longitude],zoom=9):

    map=folium.Map(location=location,control_state=True,zoom_start=zoom)

    return map

fmap=worldmap()

folium.TileLayer("cartodbpositron").add_to(fmap)

HeatMap(data=dup[["lat","long"]].groupby(["lat","long"]).sum().reset_index().values.tolist(),

       radius=8,max_zoom=13,name='Heat Map').add_to(fmap)

folium.LayerControl(collapsed=False).add_to(fmap)

fmap
plt.figure(figsize=(10,6))

sns.boxplot(x="waterfront",y="price",data=housedata)
housedata.drop(['id','zipcode'],axis=1,inplace=True)
housedata["date"].head()
housedata["date"]=pd.to_datetime(housedata["date"])

housedata["date"].head()
housedata["year"]=housedata["date"].apply(lambda date: date.year)

housedata["month"]=housedata["date"].apply(lambda date: date.month)
housedata.head()
plt.figure(figsize=(10,6))

sns.boxplot(x="month",y="price",data=housedata)
housedata.groupby("month").mean()["price"]
plt.figure(figsize=(10,6))

housedata.groupby("month").mean()["price"].plot()
plt.figure(figsize=(10,6))

housedata.groupby("year").mean()["price"].plot()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
x=housedata.drop(["price","date"],axis=1).values

y=housedata["price"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=23)

scalar=StandardScaler()

x_train=scalar.fit_transform(x_train)

x_test=scalar.transform(x_test)
model=Sequential()
def model_creating():

    model=Sequential()

    model.add(Dense(19,activation="relu"))

    model.add(Dense(19,activation="relu"))

    model.add(Dense(19,activation="relu"))

    model.add(Dense(19,activation="relu"))

    model.add(Dense(1))

    model.compile(optimizer="adam",loss="mse")

    return model
model=model_creating()
model.fit(x=x_train,y=y_train,

          validation_data=(x_test,y_test),

         batch_size=130,epochs=550,verbose=1)
model.summary()
from keras.utils.vis_utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
loss=pd.DataFrame(model.history.history)

loss.head()
loss.plot()

#if both lines are coincide then our model is not overfitting

#if we get spikes in our plot then our model is overfitting
y_pred=model.predict(x_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score
error=pd.DataFrame([[mean_squared_error(y_test,y_pred),

                     np.sqrt(mean_squared_error(y_test,y_pred)),

                    mean_absolute_error(y_test,y_pred),

                    explained_variance_score(y_test,y_pred)]],

                   columns=["mean_squared_error","mean_squared_root_error",

                                 "mean_absolute_error","explained_variance_score"])

error
print(error["mean_absolute_error"],housedata.describe()["price"]["mean"])
sample_house=housedata.drop(["price","date"],axis=1).iloc[0].values

sample_house=sample_house.reshape(-1,19)
sample_house=scalar.transform(sample_house)
sample_predict=model.predict(sample_house)

print(sample_predict,housedata.iloc[0:1,1:2].values)
plt.figure(figsize=(10,6))

plt.scatter(y_test,y_pred,color="blue",marker="o")

plt.plot(y_pred,y_pred,marker='o',

         color='green',markerfacecolor='red',

         markersize=7,linestyle='dashed')