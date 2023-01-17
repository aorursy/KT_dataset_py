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
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset=pd.read_csv("/kaggle/input/car-price-prediction/CarPrice_Assignment.csv")
print(np.size(dataset))
# print(dataset.head())
dataset.drop(columns='car_ID',inplace=True)
# print(dataset.head())
dataset.info()
print(dataset.describe().T)

# Checking for missing or inappropriate values
print((dataset.isna().sum()/dataset.shape[0])*100)
# We can also do the check by
print("Dataframe command")
print(pd.DataFrame(dataset.isnull()))

print(dataset.head())
x=dataset[['wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg']]
print(x.head())
y=dataset['price']
print(y.head())
from sklearn.model_selection import train_test_split
# Splitting the dataset
x_train,x_test,y_traian,y_test=train_test_split(x,y,test_size=0.33)
print(np.size(x_train))
print(np.size(y_traian))

# Plotting the training and testing data
# plt.figure(figsize=(20,15))
# plt.scatter(x_train['wheelbase'],y,c='r')
# plt.scatter(x_test,y_test,c='b')
# plt.show()

# intiating the model
reg=linear_model.LinearRegression()
reg.fit(x_train,y_traian)
print(reg.score(x_test,y_test))
import seaborn as sns
sns.heatmap(dataset.corr())
y_pred=reg.predict(x_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))