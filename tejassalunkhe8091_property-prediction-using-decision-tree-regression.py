# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import export_graphviz 

from sklearn.tree import export_graphviz  



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd



housedata = pd.read_csv("../input/housedata/data.csv")
housedata.columns
housedata.head(5)
housedata= housedata.dropna(axis=0)
y = housedata.price
housefeatures = ['floors','sqft_living','sqft_lot']
x = housedata[housefeatures]
x.head()
from sklearn.tree import DecisionTreeRegressor





housepricemodel = DecisionTreeRegressor(random_state=1)



housepricemodel.fit(x,y)

print("Making predictions for the following 5 houses:")

print(x.head())

print("The predictions are")

print(housepricemodel.predict(x.head(30)))
#lets take one more data set



newdata = pd.read_csv("../input/housedata/output.csv")



newdata.columns



newdata= newdata.dropna(axis=0)
newdatafeatures = ['floors','sqft_living','sqft_lot']



new = newdata[newdatafeatures]





print("Making predictions for the following 5 houses:")

print(new.head())

print("The predictions are")

print(housepricemodel.predict(new.head(30)))