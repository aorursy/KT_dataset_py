import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

plt.style.use('ggplot')

#ggplot is R based visualisation package that provides better graphics with higher level of abstraction

import os

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
diamond_data = pd.read_csv("/kaggle/input/diamonds/diamonds.csv")

diamond_data = diamond_data.drop(["Unnamed: 0"],axis=1)

diamond_data.head()
diamond_data.info()
plt.rcParams.update({"figure.figsize": (15, 10)})

sns.barplot(x=diamond_data.carat, y=diamond_data['price'])
sns.barplot(x=diamond_data.cut, y=diamond_data['price'])
sns.scatterplot(x=diamond_data['carat'], y=diamond_data['price'], hue=diamond_data['cut'])
sns.heatmap(diamond_data.corr(), annot=True,cmap='RdYlGn',square=True)
print("Number of rows with x == 0: {} ".format((diamond_data.x==0).sum()))

print("Number of rows with y == 0: {} ".format((diamond_data.y==0).sum()))

print("Number of rows with z == 0: {} ".format((diamond_data.z==0).sum()))

print("Number of rows with depth == 0: {} ".format((diamond_data.depth==0).sum()))
diamond_data[['x','y','z']] = diamond_data[['x','y','z']].replace(0,np.NaN)
diamond_data.isnull().sum()
diamond_data.dropna(inplace=True)

diamond_data.isnull().sum()
from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()



# Get list of categorical variables

s = (diamond_data.dtypes == 'object')

object_cols = list(s[s].index)



for col in object_cols:

    diamond_data[col] = label_encoder.fit_transform(diamond_data[col])

    

diamond_data.head()
sns.heatmap(diamond_data.corr(), annot=True,cmap='RdYlGn',square=True)
y = diamond_data.price

diamond_features = ['carat', 'x', 'y', 'z']

X = diamond_data[diamond_features]

X.describe()
y.describe()
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_X, train_y)



dia_preds = forest_model.predict(val_X)

print(RandomForestRegressor.score(X=train_X, y=train_y, self=forest_model))
print(mean_absolute_error(val_y, dia_preds))