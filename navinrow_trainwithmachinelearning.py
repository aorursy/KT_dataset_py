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
df = pd.read_json("/kaggle/input/wine-reviews/winemag-data-130k-v2.json")

df1=pd.read_csv("/kaggle/input/wine-reviews/winemag-data-130k-v2.csv")

df2 = pd.read_csv("/kaggle/input/wine-reviews/winemag-data_first150k.csv")
df
confirm = df[['country','points','price','province']].dropna()
features = ['points']

y = confirm.price

X = confirm[features]
from sklearn.ensemble import RandomForestRegressor
modeling = RandomForestRegressor(random_state=1,max_leaf_nodes=100)
modeling.fit(X,y)
hall = modeling.predict(X)
output1 = pd.DataFrame({'points': confirm.points,

                      'Prices based on points': hall})

output1.to_csv('random_forest_wine_prediction_based_on_points1.csv',index=False)

reader = pd.read_csv('random_forest_wine_prediction_based_on_points1.csv')

reader.head(40)