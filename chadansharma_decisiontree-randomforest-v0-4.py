# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Path of the file to read

iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)

home_data1 = home_data[home_data.SalePrice<400000]
home_data.head()
# home_data.SalePrice.hist(bins=100)

home_data1.SalePrice.hist(bins=100)
import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

# Create target object and call it y



y = home_data1.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data1[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(criterion='mse', random_state=1,max_leaf_nodes=100)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE: {:,.0f}".format(val_mae))

## Write a function to find out the best value of max leaf node
from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus

dot_data = StringIO()

export_graphviz(iowa_model, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True,feature_names=features)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
import numpy as np

import matplotlib.pyplot as plt

importances = iowa_model.feature_importances_



# Sort feature importances in descending order

indices = np.argsort(importances)[::-1]



# Rearrange feature names so they match the sorted feature importances

names = [features[i] for i in indices]



# Create plot

plt.figure()



# Create plot title

plt.title("Feature Importance")



# Add bars

plt.bar(range(X.shape[1]), importances[indices])



# Add feature names as x-axis labels

plt.xticks(range(X.shape[1]), names, rotation=90)



# Show plot

plt.show()
data=pd.DataFrame({'pred':val_predictions,'act':val_y}).sort_index(ascending=True)

data['pred']=data['pred'].astype('int')

data['err']=data.act-data.pred
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(figsize=(18, 8))



final=data.sort_values('act',ascending=False)

final = final[['act','err']]

final.plot.bar(stacked=True,ax=axarr)
data.plot.scatter('pred','act')
from sklearn.ensemble import RandomForestRegressor



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1,criterion='mse',min_samples_leaf=4)



# fit your model

rf_model.fit(train_X,train_y)

val_pred = rf_model.predict(val_X)



# Calculate the mean absolute error of your Random Forest model on the validation data

rf_val_mae = mean_absolute_error(val_pred,val_y)



print("Validation MAE for Random Forest Model: {:.0f}".format(rf_val_mae))
importances = rf_model.feature_importances_



# Sort feature importances in descending order

indices = np.argsort(importances)[::-1]



# Rearrange feature names so they match the sorted feature importances

names = [features[i] for i in indices]



# Create plot

plt.figure()



# Create plot title

plt.title("Feature Importance")



# Add bars

plt.bar(range(X.shape[1]), importances[indices])



# Add feature names as x-axis labels

plt.xticks(range(X.shape[1]), names, rotation=90)



# Show plot

plt.show()
data2=pd.DataFrame({'pred':val_pred,'act':val_y}).sort_index(ascending=True)

data2['pred']=data2['pred'].astype('int')

data2['err']=data2.act-data2.pred
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(figsize=(18, 8))



final2=data2.sort_values('act',ascending=False)

final2 = final2[['act','err']]

final2.plot.bar(stacked=True,ax=axarr)
data2.plot.scatter('pred','act')
a= home_data.isna().mean().round(4) * 100

a[a.values>0]
y = home_data.SalePrice

X = home_data.iloc[:,:-1]

X = X.drop(columns=['Alley','Fence','FireplaceQu','MiscFeature','PoolQC'])

X = X.fillna(0)

X = pd.get_dummies(X)

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rf_model = RandomForestRegressor(random_state=1,criterion='mse',min_samples_leaf=4)



# fit your model

rf_model.fit(train_X,train_y)

val_pred = rf_model.predict(val_X)



# Calculate the mean absolute error of your Random Forest model on the validation data

rf_val_mae = mean_absolute_error(val_pred,val_y)



print("Validation MAE for Random Forest Model: {:.0f}".format(rf_val_mae))
features = X.columns.values

importances = rf_model.feature_importances_

# Sort feature importances in descending order

indices = np.argsort(importances)[::-1]



# Rearrange feature names so they match the sorted feature importances

names = [features[i] for i in indices]



# Create plot

plt.figure(figsize=(50,10))



# Create plot title

plt.title("Feature Importance")



# Add bars

plt.bar(range(X.shape[1]), importances[indices])



# Add feature names as x-axis labels

plt.xticks(range(X.shape[1]), names, rotation=90)



# Show plot

plt.show()
data3=pd.DataFrame({'pred':val_pred,'act':val_y}).sort_index(ascending=True)

data3['pred']=data3['pred'].astype('int')

data3['err']=data3.act-data3.pred
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(figsize=(18, 8))



final3=data3.sort_values('act',ascending=False)

final3 = final3[['act','err']]

final3.plot.bar(stacked=True,ax=axarr)
data3.plot.scatter('pred','act')