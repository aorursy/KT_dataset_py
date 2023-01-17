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
import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")

display(df.head(3))

df.columns
#MULTI VARIATE



features = ['bedrooms', 'bathrooms', 'sqft_living',

       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',

       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',

       'lat', 'long', 'sqft_living15', 'sqft_lot15']

X = df[features]

y = df["price"]

X = np.c_[np.ones(X.shape[0]),X]

print(X.shape,y.shape)
#calculating theta directly without gradient descent



theta = np.dot(np.linalg.pinv(np.dot(X.T,X)),np.dot(X.T,y))
theta.shape
theta
predict = np.dot(X,theta)

predict