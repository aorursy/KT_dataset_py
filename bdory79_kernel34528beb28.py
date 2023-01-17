#Badriah Alsaeedi 1676021

#Renad Alharbi 1612332


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# me = melbourn

mepath = '../input/melbourne-housing-snapshot/melb_data.csv'

mehData = pd.read_csv(mepath)



mehData.info()

mehData.describe()

mehData.corr()

mehData.dropna(axis=0)

mehData.info()



y = mehData.Price

meFeatures = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

x = mehData[meFeatures]

#y.head()





from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)



from sklearn.tree import DecisionTreeRegressor

meModel = DecisionTreeRegressor(random_state = 1)

meModel.fit(train_x, train_y)



from sklearn.metrics import mean_absolute_error

val_predictions = meModel.predict(val_x)

print(mean_absolute_error(val_y, val_predictions))
