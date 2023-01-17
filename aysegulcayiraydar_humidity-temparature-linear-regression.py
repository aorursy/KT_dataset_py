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
data = pd.read_csv('/kaggle/input/szeged-weather/weatherHistory.csv')
data.info()
humidity = data[['Humidity']]

temparature = data[['Temperature (C)']]

data_new = pd.concat([ humidity,temparature],axis=1)

data_new.corr()
from sklearn.model_selection import train_test_split



x = humidity['Humidity'].values.reshape(-1,1)



y = temparature['Temperature (C)'].values.reshape(-1,1)



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)
import matplotlib.pyplot as plt



#x_train=x_train.sort_index()

#y_train=y_train.sort_index()



from sklearn.linear_model import LinearRegression



linear_reg = LinearRegression()



linear_reg.fit(x_train,y_train)



# visualize line



plt.scatter(x_train,y_train)

y_head = linear_reg.predict(x_test)

plt.xlabel("humidity")

plt.ylabel("temparature")

plt.plot(x_test , y_head , color = "green")

plt.show()