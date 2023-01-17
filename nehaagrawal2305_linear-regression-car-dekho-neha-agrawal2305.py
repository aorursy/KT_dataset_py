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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score
cars = pd.read_csv('/kaggle/input/car-dekho-data/car data.csv')

cars
labelencoder=LabelEncoder()
cars_name = labelencoder.fit_transform(cars['Car_Name'])

fueltype = labelencoder.fit_transform(cars['Fuel_Type'])

Selltype = labelencoder.fit_transform(cars['Seller_Type'])

trans = labelencoder.fit_transform(cars['Transmission'])
cars['Car_Name']=cars_name

cars['Fuel_Type']=fueltype

cars['Seller_Type'] = Selltype

cars['Transmission']=trans
cars.info()
x=cars.drop(['Selling_Price','company'],axis = 1)

y = cars[['Selling_Price']].values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state=10)

x_test.shape
linear = LinearRegression()
car_model = linear.fit(x_train, y_train)
predict = linear.predict(x_test)
predict
score = np.abs(np.mean(y_test-predict))

score*100
sns.pairplot(cars)
sns.scatterplot('Year','Selling_Price',data=cars,hue='Owner',style='Fuel_Type')
fig = plt.figure()

ax = fig.gca(projection='3d')

ax.plot_trisurf(cars['Year'], cars['Selling_Price'], cars['Fuel_Type'], cmap=plt.cm.jet, linewidth=0.01)

# ax.view_init(30,45)

plt.xlabel('Overall Condition')

plt.ylabel('Year')

plt.show()