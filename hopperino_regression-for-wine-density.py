#import the data

import pandas as pd



wine = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
#quick check the data

wine.head()
#check datatypes

wine.info()
#look at basic statistics

wine.describe()
#quick plot the data

%matplotlib inline

import matplotlib.pyplot as plt

wine.hist(bins=50,figsize=(20,15))

plt.show()
import numpy as np



np.random.seed(42) #makes it reproducable
#divide into train and test

from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(wine, test_size=0.2, random_state=42)

wine = train_set.copy() #create a copy to play with
import seaborn as sns



sns.pairplot(wine) #pairplot to look at correlations
#check correlations

sns.heatmap(wine.corr()) 
#look at a specific correlation

sns.scatterplot(x='fixed acidity',y='density',data=wine)
#separate data into data and labels

wine_x = train_set.drop("density",axis=1)

wine_y = train_set["density"].copy()
#create regression model

from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(wine_x, wine_y)
#apply regression model

predictions = lin_reg.predict(wine_x)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(wine_y, predictions))

print('MSE:', metrics.mean_squared_error(wine_y, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(wine_y, predictions)))
plt.title('Comparison of Y values in test and the Predicted values')

plt.ylabel('Test Set')

plt.xlabel('Predicted values')

plt.scatter(predictions, wine_y,  color='black',alpha=0.1)

plt.show()