



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings        

warnings.filterwarnings('ignore')

housing = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv');

housing.head()

#the dataset is made of 12 features, and we want to predict the median_house_value, for this reason we ignore latitude and longitude because are irrelevant for value prediction
#let's observe the data

housing.hist(bins=50, figsize=(20,20))

plt.show()
# fill empty parameters with a mean value 

housing['total_bedrooms'][housing['total_bedrooms'].isnull()] = np.mean(housing['total_bedrooms'])

housing.loc[290] #access values by index (DataFrame.loc)

#encode categorical values of ocean_proximity feature

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



labelencoder = LabelEncoder();

labelenc = labelencoder.fit_transform(housing['ocean_proximity']);

housing['ocean_proximity'] = labelenc;

housing.head()
#convert categorical features with onehotencoding

onehotenc = OneHotEncoder(sparse = False);

onehot = onehotenc.fit_transform(np.array(housing['ocean_proximity']).reshape(-1,1));

housing['NEAR BAY'] = onehot[:,0];

housing['INLAND'] = onehot[:,1];

housing['<1H OCEAN'] = onehot[:,2];

housing['ISLAND'] = onehot[:,3];

housing['NEAR OCEAN'] = onehot[:,4];

datasetfixed = housing.drop('ocean_proximity', axis=1);



datasetfixed
#let's check the correlation between features

import seaborn as sns



corrmat = datasetfixed.corr()

plt.figure(figsize=(10,10))

g = sns.heatmap(corrmat,annot=True,cmap="RdYlGn")
#apply linear regression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



X = datasetfixed.drop('median_house_value', axis= 1);

Y = datasetfixed['median_house_value'];



trainX,testX,trainY,testY = train_test_split(X, Y, test_size=0.3)



clf = LinearRegression()

clf.fit(np.array(trainX),trainY)



#use mean squared error as accuracy evaluator

predictions = clf.predict(testX)

mse = mean_squared_error(testY, predictions)

rmse = np.sqrt(mse)

rmse
