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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline 
data = pd.read_csv("../input/fish-market/Fish.csv")
data.head()
data.isnull().sum()
data['Species'].value_counts()
data =pd.get_dummies(data)
data.columns
X=data[['Length1', 'Length2', 'Length3', 'Height', 'Width',

       'Species_Bream', 'Species_Parkki', 'Species_Perch', 'Species_Pike',

       'Species_Roach', 'Species_Smelt', 'Species_Whitefish']]
y = data['Weight']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)
y_predict = regression.predict(X_test)
output = pd.DataFrame({"actual":y_test,"predicted":y_predict})

output
plt.scatter(y_test,y_predict)