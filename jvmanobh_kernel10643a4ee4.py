# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ams = pd.read_json('../input/amsterdam.json').sort_index()
#Answer:

ams.shape[0]
ams.columns
#Answer: Yes

ams['bedrooms'].isnull().sum()
#Answer: Yes, It's a numerical variable and it belongs to float data type

ams.info()
# Price is a character datatype, since it contains currency symbol $ and ',' Removing the special characters and converting it to numerical

ams['price'] = ams['price'].apply(lambda x: x.replace('$','').replace(',','')).astype('float')
#Answer: 134.799

round(ams['price'].mean(),3)
#Answer: 3142

round(ams['price'].max(),1)
#Answer: 440

round(ams['price'].quantile(0.99),1)
# Removing all rows with price greater than 440

ams1 = ams[~(ams['price']>440)].copy()
#Answer: 210

round(ams1['price'].quantile(0.9),1)
# Ans: 14998

ams1.dropna(inplace=True)

ams1.shape[0]
#Defining function for calculating distance from city center

def dist_calc(lon1, lat1, lon2=4.899431, lat2=52.379189):

    """

    Calculate the great circle distance between two points

    on the earth (specified in decimal degrees)



    All args must be of equal length.    



    """

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2



    c = 2 * np.arcsin(np.sqrt(a))

    km = 6367 * c

    return km
#Creating d_centre

ams1['d_centre'] = ams1.apply(lambda x: dist_calc(x['longitude'], x['latitude']), axis=1)
#Answer: 41

round(ams1.sort_values(by='d_centre')['price'],1).head()
#Answer: 118.1

round(ams1[(ams1['d_centre']<=1) & (ams1['room_type']=='Private room')]['price'].mean(), 1)
#Answer: 14 columns

#Creating Predictor X and Response Matrix y

X = pd.get_dummies(ams1.drop(['price' ,'latitude', 'longitude'], axis=1), drop_first=True)

y = ams1['price']

X.info()
#Answer: 0.003

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=200, test_size=0.3)

round(np.abs(X_train['accommodates'].mean() - X_test['accommodates'].mean()) ,3)
rf = RandomForestRegressor(random_state=200, max_features='sqrt', oob_score=True)

grid = GridSearchCV(rf, param_grid={'n_estimators': [30, 40, 50, 100, 200, 300, 400, 500, 550, 600]}, cv=3)

grid.fit(X, y)

print(grid.best_estimator_, grid.best_score_)
#Answer: accommodates, bedrooms, d_centre

rf = RandomForestRegressor(random_state=200, max_features='sqrt', oob_score=True, n_estimators=400)

rf.fit(X_train, y_train)

pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
pred = rf.predict(X_test)

error = np.abs(y_test-pred)

error.describe()
sns.scatterplot(y_test,error)

plt.xlim(0,300)