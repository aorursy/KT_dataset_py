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
import pandas as pd #data manipulation

import numpy as np #numerical python for statistical computing

df =pd.read_csv("../input/heights-and-weights/data.csv") #reading data

df.head() #for first five rows

df.keys() #columns name
df.columns #columns name
x = df.Height #initializing x

y = df.Weight #initializing y

x = df.iloc[:,0:1].values #indexing

y = df.iloc[:,1].values #indexing

from sklearn.linear_model import LinearRegression #machine learning and statistical modelling

MachineBrain = LinearRegression() #statistical function

MachineBrain.fit(x,y)
m = MachineBrain.coef_ #slope

c = MachineBrain.intercept_ #intercept

y_predict = m*x+c #linear regression

y_predict
import matplotlib.pyplot as plt #plotting library and extension python method

plt.scatter(x,y) #scatter plot

plt.plot(x,y_predict,c="green") #line plot