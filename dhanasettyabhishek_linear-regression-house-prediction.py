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
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline





from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn import metrics
data = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")

print("Dimensions:",data.shape)

data.describe()
def checkNull(data):

    return data.isnull().any()

def checkDatatype(data):

    return data.dtypes

print(checkNull(data))

print("=================")

print(checkDatatype(data))
# Coverting date datatype from object to datetime

data['date'] =  pd.to_datetime(data['date'], format='%Y%m%dT000000')

# Pariwise plot for features such as squarefeet, price, and the number of bedrooms to see how the features are distributed

sns.pairplot(data[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], hue='bedrooms')
##### Finding the correlation between features, using heatmap

featureCorr = data.corr()

fig, ax = plt.subplots(figsize=(12,12))

sns.heatmap(featureCorr, xticklabels=featureCorr.columns, yticklabels=featureCorr.columns, ax=ax)
# From the above heatmap we can say that the columns id and date are not corelated to the decision, hence eliminating id and date column from the dataset.

new_data = data.drop(['id', 'date'], axis=1)
# In the model features are 'X' and the class/varaible we are predicting is 'y'

y = new_data.price.values

data_without_price = new_data.drop(['price'], axis=1)

X = data_without_price.values

features = data_without_price.columns

print(features,"\n", X,"\n", y)
# Spliting the dataset into training and testing.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)



# Linear Regression model

model = LinearRegression()

model.fit(X_train, y_train)
y_predict = model.predict(X_test)

y_predict
# Backward Elimination we eliminate the features based on p-Value



import statsmodels.api as sm



def backwardElimation(data, threshold):

    zeros = np.zeros(data.shape).astype(int)

    no_of_features = len(data[0])

    for i in range(no_of_features):

        model = sm.OLS(y, data).fit()

        max_value = max(model.pvalues).astype(float)

        adjR = model.rsquared_adj.astype(float)

        if max_value > threshold:

            for j in range(no_of_features-i):

                if model.pvalues[j].astype(float) == max_value:

                    zeros[:, j] = data[:, j]

                    data = np.delete(data, j, 1)

                    new_model = sm.OLS(y, data).fit()

                    new_adjR = new_model.rsquared_adj.astype(float)

                    if adjR >= new_adjR:

                        new_data = np.hstack((data, zeros[:, [0, j]]))

                        new_data = np.delete(new_data, j, 1)

                        print(model.summary())

                        return new_data

                    else:

                        continue

    model.summary()

    return data



threshold = 0.05

backwardElimation(data_without_price.values, threshold)

# Cross validation score between 'Living area square feet and price of the house'

score_train = model.score(X_train, y_train)

score_test = model.score(X_test, y_test)

print("Training score:", score_train)

print("Testing score:", score_test)

# Cross validation

crossValidation = cross_val_score(model, data[['sqft_living']], data[['price']], cv=5).mean()

print("Crossvalidation score between the living are square feet and price is:",crossValidation)
# Training the model based on the training dataset of square feet. 

sqft_living_train = X_train[:, 2]

sqft_living_test = X_test[:, 2]

bedrooms_train = X_train[:, 0]

bedrooms_test = X_test[:, 0]

sqft_living_model = LinearRegression()

sqft_living_model.fit(sqft_living_train.reshape(-1, 1), y_train)
# Regression plot

plt.scatter(sqft_living_test,y_test,label="Price Data", alpha=.3)

plt.plot(sqft_living_test,sqft_living_model.predict(sqft_living_test.reshape(-1, 1)),color="red",label="Predicted Regression Line")

plt.xlabel("Living room square feet")

plt.ylabel("Price")

plt.legend()
# Comparing price with respect to number of bedrooms

crossValidation = cross_val_score(model, data[['bedrooms']], data[['price']], cv=5).mean()

print("Crossvalidation score between the living are square feet and price is:",crossValidation)

fig, ax = plt.subplots(figsize=(12, 8))

# sns.despine(left=True, bottom=True)

sns.boxplot(x=data['bedrooms'],y=data['price'], ax=ax)

ax.yaxis.tick_left()

ax.set(xlabel='Bedrooms', ylabel='Price');
# Heatmap of King County, with respective house prices.

from mpl_toolkits.mplot3d import Axes3D

import folium

from folium.plugins import HeatMap



max_price=data.loc[data['price'].idxmax()]



def generateBaseMap(default_location=[max_price['lat'], max_price['long']]):

    base_map = folium.Map(location=default_location, control_scale=True)

    return base_map



data['count'] = 1

basemap = generateBaseMap()

s=folium.FeatureGroup(name='icon').add_to(basemap)

folium.Marker([max_price['lat'], max_price['long']],popup='Highest Price:'+str(max_price['price']),

              icon=folium.Icon(color='red')).add_to(s)

HeatMap(data=data[['lat','long', 'count']].groupby(['lat','long']).sum().reset_index().values.tolist(),

        radius=8,name='Heat Map').add_to(basemap)

folium.LayerControl(collapsed=False).add_to(basemap)

basemap
# We are able to predict the price based on a input dataset

input_data = float(input("Enter your desired square feet"))

test_data = np.array([input_data]).reshape(-1, 1)

result = sqft_living_model.predict(test_data)

print(f'The price of a house for {test_data[0][0]} square feet in Kings County, USA is: ${ round(result[0], 2) }')