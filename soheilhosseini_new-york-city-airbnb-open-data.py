"""Context

Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. This dataset describes the listing activity and metrics in NYC, NY for 2019.



Content

This data file includes all needed information to find out more about hosts, geographical availability, necessary metrics to make predictions and draw conclusions.



Acknowledgements

This public dataset is part of Airbnb, and the original source can be found on this website.



"""
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
#New York City

from PIL import Image 

img = Image.open("../input/new-york-city-airbnb-open-data/New_York_City_.png")

img
# Importing the Libraries¶



import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LinearRegression



from sklearn.linear_model import Ridge



from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings('ignore')



from sklearn.ensemble import RandomForestRegressor



from sklearn.preprocessing import OneHotEncoder



from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

import plotly.express as px



import statsmodels.api as sm

from statsmodels.formula.api import ols



from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR



from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score

# Load AirBnB Data

ab = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
# Create a copy of data

# different between copy and deepcopy in python

ab_copy =ab.copy()
ab.head()
ab.info()
ab.describe()
ab.describe(include='O')
#Find out missing values in Dataset

np.sum(pd.isnull(ab))
#Missing values

#How many missing values are there for each feature?  



missing_values = pd.isnull(ab).sum()/len(ab)

miss_values = pd.DataFrame(data = missing_values,index = ab.columns, columns =['missing values per column'])



miss_val = miss_values.sort_values(by='missing values per column')

plt.figure(figsize=(7,7))

sns.barplot(x='missing values per column',y=miss_val.index, data =miss_val)

plt.show()
plt.title('Share in Neighborhood',fontsize=20)

sns.countplot(x='neighbourhood_group', data = ab)

plt.show()
fig = px.pie(ab,  names='neighbourhood_group', title='Share in Neighborhood')

fig.show()
plt.figure(figsize=(6,6))

sns.scatterplot(x="longitude", y="latitude", data=ab, hue='neighbourhood_group',alpha=0.2)

plt.show()
ab.availability_365.head()
ab.loc[ab['price']<500].plot.scatter(x='longitude',

                y='latitude',

                c='price',

                colormap ='jet',alpha=0.4)

plt.show()
ab[['neighbourhood', 'price']].groupby(['neighbourhood'], as_index=False).median().sort_values(by='price', ascending=False)[0:10]
plt.title('Room_type counts',fontsize=20)

sns.countplot(x='room_type', data = ab)

plt.show()
fig = px.pie(ab,  names='room_type', title='Share in room_type')

fig.show()
plt.title('Room type in different neighborhoods',fontsize=20)

sns.countplot(x='neighbourhood_group', hue ='room_type' , data = ab)

plt.show()
plt.figure(figsize=(6,6))

sns.distplot(ab.loc[ab['minimum_nights']<30]['minimum_nights'], color='red')

plt.xticks(np.arange(min(ab['minimum_nights']), 31, 1), rotation=90)



plt.show()
plt.ylabel('density')



sns.distplot(ab['price'], color='red')

plt.show()
#price distribution with natural log

plt.figure(figsize=(6,6))

plt.ylabel('density')

sns.distplot(np.log10(1+ ab['price']), color='purple')

plt.show()
plt.title("neighborhood group box plot")

sns.boxplot(x="neighbourhood_group", y="price", data=ab.loc[ab['price']<500])

plt.show()
sns.boxplot(x="room_type", y="price", data=ab_copy[ab_copy['price']<500])

plt.show()
plt.figure(figsize=(6,6))

plt.title("Relationship between number of reviews and price")

sns.scatterplot(x = 'number_of_reviews',y = 'price',data = ab,color='red')

plt.show()
#Correlation Matrix

fig = plt.figure(figsize=[7,7])

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(ab.corr(), annot = True, square=True,linecolor='white',cmap='coolwarm',vmin=-1, vmax = 1 )

plt.show()
#what is collinearity??

# cardinality (decrease cardinality of categorical variables)

# categorical correlation
ab.columns
ab.describe()
# Remove observations with price equal to 0 (faulty records). 

ab = ab[ab['price']!=0]
ab.head()
#Remove redundant features

ab.drop(columns = ['id', 'name', 'host_id', 'host_name','last_review'], inplace=True)
# Check out numbers of duplicated rows

np.sum(ab.duplicated())
#Wherever last_review and reviews_per_month are missing at the same times.

ab.fillna(0, inplace=True)

np.sum(pd.isnull(ab))
#Change categorical variables to numeric ones

ab = pd.get_dummies(data = ab, drop_first = True)
#Scale variables



scaler = StandardScaler()

ab_scaled = scaler.fit_transform(ab)
ab_scaled =pd.DataFrame(data = ab_scaled, columns = ab.columns)
ab_scaled.shape
X = ab_scaled.drop(columns = "price")

y = ab_scaled["price"]
X.shape
#Feature selection



from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestRegressor

selector = SelectFromModel(estimator=RandomForestRegressor(n_jobs=-1, n_estimators=100)).fit(X, y)

X_new = selector.transform(X)
X_new.shape
#Convert array to dataframe

X_new = pd.DataFrame(data= X_new)
X_train, X_test, y_train, y_test = train_test_split(

    X_new, y, test_size=0.2, random_state=42)
X_train.head()
model ={'Decision Tree':DecisionTreeRegressor(max_depth=2,min_samples_leaf =10, random_state=0),

        

       'Random Forest Classifier': RandomForestRegressor( n_jobs=-1, n_estimators=500, max_depth=2, random_state=0),

       

       'Gradient Boosting': GradientBoostingRegressor(  n_estimators=500,max_depth=2,

                                 min_samples_leaf=5)}
for keys, items in model.items():



    print(f"cross validation scores: {keys.upper()} ")

    print(cross_val_score(items, X_new, np.log1p(y), cv=5),"\n")
#R-squared is 0.095, which is not very good.
from keras import backend as K



# Is this computing the right thing?

def det_coeff(y_true, y_pred):

    u = K.sum(K.square(y_true - y_pred))

    v = K.sum(K.square(y_true - K.mean(y_true)))

    return K.ones_like(v) - (u / v)
from keras import models

from keras import layers

from keras import regularizers





def build_model():

    # Because we will need to instantiate

    # the same model multiple times,

    # we use a function to construct it.

    model = models.Sequential()

    model.add(layers.Dense(128, activation='relu',input_shape=(X_train.shape[1],)))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dropout(0.5))



    

    model.add(layers.Dense(1))

    model.compile(optimizer='RMSprop', loss='mse', metrics=[det_coeff])

    return model
model = build_model()

history = model.fit(X_train, np.log1p(y_train),  epochs=20, batch_size=200, verbose=1)
test_mse_score, test_mae_score = model.evaluate(X_test, np.log1p(y_test))

test_mae_score