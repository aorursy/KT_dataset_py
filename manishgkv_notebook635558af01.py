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
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import folium

from folium import plugins



plt.rcParams['figure.figsize']=10,12



import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
dataset=pd.read_excel('../input/same-as/Onion_2019.xlsx')
dataset.drop('Arrival Date', axis=1, inplace=True)

dataset
dataset.describe()
x=dataset[['Arrivals (Tonnes)','Minimum Price(Rs./Quintal)','Maximum Price(Rs./Quintal)']].values

y=dataset['Modal Price(Rs./Quintal)'].values
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print(regressor.coef_)
print(regressor.intercept_)
predicted=regressor.predict(x_test)
print(predicted)
dframe= pd.DataFrame({'Actual': y_test.flatten(),'Predicted':predicted.flatten()})
dframe.head(25)
import math
print('mean absolute error', metrics.mean_absolute_error(y_test,predicted))

print('mean squared error', metrics.mean_squared_error(y_test,predicted))

print('root mean squared error', math.sqrt(metrics.mean_absolute_error(y_test,predicted)))
graph=dframe.head(20)
graph.plot(kind='bar')