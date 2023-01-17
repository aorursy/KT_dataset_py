# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Loading data
path = '../input/chopstick-effectiveness.csv'
data = pd.read_csv(path)


# Check data
data.head()
# Check correlation between features
sns.pairplot(data)
# Create X and Y
necessary_features = ['Food.Pinching.Efficiency', 'Individual']
X = data[necessary_features]
Y = data[['Chopstick.Length']]
print(Y.head())
# Split data
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33)
print(len(train_Y))
# Learn algorithm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def choose_better_model(amount_leafes):
    model = DecisionTreeRegressor(max_depth=amount_leafes)
    model.fit(train_X, train_Y)
    predicted_value = model.predict(test_X)
    return model, mean_absolute_error(predicted_value, test_Y)

best_score = 0
for i in [1, 2, 3, 4, 5]:
    model, score = choose_better_model(i)
    print(score)
    if score > best_score:
        final_model = model
        best_score = score
        
