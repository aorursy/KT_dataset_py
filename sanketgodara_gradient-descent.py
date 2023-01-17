import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import re
Bengaluru_House_Data = pd.read_csv("../input/bengaluru-house-price-data/Bengaluru_House_Data.csv")
Bengaluru_House_Data.head()
y = Bengaluru_House_Data["price"]
Bengaluru_House_Data["area_type"].unique()

area_dict = {'Super built-up  Area':0, 'Plot  Area':1, 'Built-up  Area':2,'Carpet  Area':3}
Bengaluru_House_Data['area_type_code'] = Bengaluru_House_Data.apply(

    lambda row: area_dict[row['area_type']],

    axis=1

)
Bengaluru_House_Data.drop(["society","availability"], axis=1,inplace=True )
Bengaluru_House_Data.dropna(inplace=True)
Bengaluru_House_Data.head()
Bengaluru_House_Data['area'] = Bengaluru_House_Data.apply(

    lambda row: re.match(r"\d+", row["total_sqft"]).group(0),

    axis=1

)
Bengaluru_House_Data['rooms'] = Bengaluru_House_Data.apply(

    lambda row: int(row["size"].split()[0]),

    axis=1

)
Bengaluru_House_Data['new_location'] = Bengaluru_House_Data.apply(

    lambda row: str(row['location']).lower().replace(" ", "") ,

    axis=1

)
#Normalize area

mean_area = Bengaluru_House_Data["area"].mean()

std_area = Bengaluru_House_Data["area"].std()

def hypothesis(X, weights):

    return np.dot(X,weights)
def cost_fn(weights, X, y):

    cost = 0

    for idx, y_i in enumerate(y):

        cost = cost + (hypothesis(X[idx],weights) - y_i)*(hypothesis(X[idx],weights) - y_i)

    return cost
def dcost(j, weights, X, y):

    cost = 0

    for idx, y_i in enumerate(y):

        cost = cost + (hypothesis(X[idx],weights) - y_i)*X[idx][j]

    return cost
def gradient_descent(weights, X, y, lr, hypotesis):

    new_wts = weights

    for j in range(len(weights)):

        new_wts[j] = new_wts[j] - lr*dcost(j, weights, X, y)

    weights = new_wts

    return weights
X = [[1,5,5,1,9],[1,6,3,2,8],[1,4,3,2,6],[1,8,2,1,7]]

weights=[1,1,1,1,1]
y = [460,315, 232,178]
gradient_descent(weights,X,y,0.01,hypothesis)

plt.plot(y, hypothesis(X,weights), marker="o")

print(cost_fn(weights, X,y))