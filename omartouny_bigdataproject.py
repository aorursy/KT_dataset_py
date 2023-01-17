# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
import seaborn as sns 
import matplotlib.pyplot as plt 
import re 
from mpl_toolkits.basemap import Basemap 
import sklearn
%matplotlib inline

# preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
import pandas_profiling as pp

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error,r2_score



df = pd.read_csv('../input/craigslist-carstrucks-data/vehicles.csv')
df.head()

print(len(df.columns))
print(df.columns)
print(df.size)
print(df.shape)
nulls = df.isnull().sum().sort_values(ascending=False).div(len(df))
plt.figure(figsize=(24,12))
sns.barplot(x=nulls.index, y=nulls.values)
plt.title("Percent of missing data")
plt.show()
data = df.drop(["county","paint_color","description","image_url","size"],axis=1)
print(data.size)
print(data.shape)
print(len(data.columns))
data.head()
data.info()
def cylinders_adjustment(row):
    if type(row["cylinders"]) is str:
        cyl = re.findall(r"(\d) cylinders", row["cylinders"])
        if len(cyl) != 0:
            return int(cyl[0])
        else:
            return -1
    else:
        return -1
        
#data["cylinders"] = data.apply(cylinders_adjustment,axis=1)
def condition_adjustment(row):
    if type(row["condition"]) is str:
        if(row["condition"]=='excellent'):
            return 0
        elif(row["condition"]=='good' or row["condition"]=='fair'):
            return 1
        elif(row["condition"]=='new'or row["condition"]=='like new'):
            return 2
        else:
            return -1
    else:
        return -1
        
data = data.dropna()
print(data.shape)
data.info()
data_copy= data.copy()
data_copy.shape
manufacturers = data_copy["manufacturer"].value_counts().div(len(data_copy)).mul(100)
manufactuters_TOP20 = manufacturers[:20]

plt.figure(figsize=(16,8))
sns.barplot(x=manufactuters_TOP20.index, y=manufactuters_TOP20.values)
plt.title("20 most popular manufactureres in the USA")
plt.ylabel("Popularity in %")
plt.xticks(rotation=90)
plt.show()
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = data.columns.values.tolist()
for col in features:
    if data[col].dtype in numerics: continue
    categorical_columns.append(col)
# Encoding categorical features
for col in categorical_columns:
    if col in data.columns:
        le = LabelEncoder()
        le.fit(list(data[col].astype(str).values))
        data[col] = le.transform(list(data[col].astype(str).values))
data.info()
data["year"] = data["year"].astype("int64")
data["odometer"] = data["odometer"].astype("int64")
data["lat"] = data["lat"].astype("int64")
data["long"] = data["long"].astype("int64")
data.info()
corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')
data.describe()
pp.ProfileReport(data)
data = data[data['price'] > 1000]
data = data[data['price'] < 375000]
data = data[data['model'] > 0]
plt.figure(figsize=(20,10))

data_sub = data.sample(80000)

m = Basemap(projection='merc', # mercator projection
            llcrnrlat = 20,
            llcrnrlon = -170,
            urcrnrlat = 70,
            urcrnrlon = -60,
            resolution='l')

m.shadedrelief()
m.drawcoastlines() # drawing coaslines
m.drawcountries(linewidth=2) # drawing countries boundaries
m.drawstates(color='b') # drawing states boundaries
#m.fillcontinents(color='grey',lake_color='aqua')

for index, row in data_sub.iterrows():
    latitude = row['lat']
    longitude = row['long']
    x_coor, y_coor = m(longitude, latitude)
    m.plot(x_coor,y_coor,'.',markersize=0.9,c="red")
plt.figure(figsize=(16,9))
sns.boxplot(x="year", y="price", data = data)
plt.title("Price of cars vs. manufacturing year")
plt.ylabel("Price [USD]")

max_year = data["year"].max()
min_year = data["year"].min()
steps = 2
lab = np.sort(data["year"].unique())[::2]
pos = np.arange(0,111,2)

plt.xticks(ticks=pos, labels=lab, rotation=90)
plt.show()
train_target = data["price"]
data_ = data.drop(columns=["price"])
train0, test0, train_target0, test_target0  = train_test_split(data_, train_target, test_size=0.2, random_state=0)
#For models from Sklearn
scaler = StandardScaler()
train0 = pd.DataFrame(scaler.fit_transform(train0), columns = train0.columns)
train, test, target, target_test = train_test_split(train0, train_target0, random_state=0)
print(train.info())
print(test.info())
def acc_model(model,train,test):
    # Calculation of accuracy of model акщь Sklearn by different metrics   
    
    ytrain = model.predict(train)  
    ytest = model.predict(test)

    acc_train_r2_num = round(r2_score(target, ytrain) * 100, 2)
    print('acc(r2_score) for train =', acc_train_r2_num)   

    
    acc_test_r2_num = round(r2_score(target_test, ytest) * 100, 2)
    print('acc(r2_score) for test =', acc_test_r2_num)
    

random_forest = RandomForestRegressor()
random_forest.fit(train, target)
acc_model(random_forest,train,test)