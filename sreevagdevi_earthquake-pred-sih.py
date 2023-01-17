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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
data = pd.read_csv("../input/database.csv")

data
import pandas as pd

database = pd.read_csv("../input/database.csv")
data.columns
data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

data.head()
import datetime

import time



timestamp = []

for d, t in zip(data['Date'], data['Time']):

    try:

        ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')

        timestamp.append(time.mktime(ts.timetuple()))

    except ValueError:

        # print('ValueError')

        timestamp.append('ValueError')
timeStamp = pd.Series(timestamp)

data['Timestamp'] = timeStamp.values

data['Timestamp']
final_data = data.drop(['Date', 'Time'], axis=1)

final_data = final_data[final_data.Timestamp != 'ValueError']

final_data.head()
from mpl_toolkits.basemap import Basemap



m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')



longitudes = data["Longitude"].tolist()

latitudes = data["Latitude"].tolist()

#m = Basemap(width=12000000,height=9000000,projection='lcc',

            #resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.)

x,y = m(longitudes,latitudes)
x
y
fig = plt.figure(figsize=(12,10))

plt.title("All affected areas")

m.plot(x, y, "o", markersize = 2, color = 'blue')

m.drawcoastlines()

m.fillcontinents(color='coral',lake_color='aqua')

m.drawmapboundary()

m.drawcountries()

plt.show()
X = final_data[['Timestamp', 'Latitude', 'Longitude']]

y = final_data[['Magnitude', 'Depth']]
X
y
import numpy as np

import sklearn

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, X_test.shape)
from sklearn.ensemble import RandomForestRegressor



reg = RandomForestRegressor(random_state=42)

reg.fit(X_train, y_train)

reg.predict(X_test)
reg.score(X_test, y_test)
#from sklearn.model_selection import GridSearchCV



#parameters = {'n_estimators':[10, 20, 50, 100, 200, 500]}



#grid_obj = GridSearchCV(reg, parameters)

#grid_fit = grid_obj.fit(X_train, y_train)

#best_fit = grid_fit.best_estimator_

#best_fit.predict(X_test)
#best_fit.score(X_test, y_test)
#[test_loss, test_acc] = model.evaluate(X_test, y_test)

#print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
print(reg.predict([[-1.57631e+08,17.123,73.45]]))