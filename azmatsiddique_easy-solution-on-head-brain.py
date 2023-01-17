#import library

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

import pandas as pd

import pandas_profiling as npp

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('/kaggle/input/headbrain/headbrain.csv')

print(data.shape)

data.head()
# Collecting X and Y

X = data['Head Size(cm^3)'].values

Y = data['Brain Weight(grams)'].values
#finding the correlation 

corr = data.corr()

#Plot figsize

fig, ax = plt.subplots(figsize=(10, 8))

#Generate Heat Map, allow annotations and place floats in map

sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")

#Apply xticks

plt.xticks(range(len(corr.columns)), corr.columns);

#Apply yticks

plt.yticks(range(len(corr.columns)), corr.columns)

#show plot

plt.show()
#scatter plot

import plotly.express as px

fig = px.scatter(data, x=data["Head Size(cm^3)"], y=data["Brain Weight(grams)"])

fig.show()
mean_x = np.mean(X)

mean_y = np.mean(Y)
# Total number of values

m = len(X)

m
# Cannot use Rank 1 matrix in scikit learn

X = X.reshape((m, 1))
# Creating Model

reg = LinearRegression()

# Fitting training data

reg = reg.fit(X, Y)

# Y Prediction

Y_pred = reg.predict(X)
# Calculating RMSE and R2 Score

mse = mean_squared_error(Y, Y_pred)

rmse = np.sqrt(mse)

r2_score = reg.score(X, Y)
print(np.sqrt(mse))

print(r2_score)