import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score

from sklearn.multioutput import MultiOutputRegressor

import matplotlib.pyplot as plt

from math import sqrt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/BodyFat.csv')
df.head()
isinstance(df, pd.DataFrame)
df.describe()
myself = [21,198.42,64.17,43,109,111,105,59,41,25.5,33.5,30,19]

myself = np.array(myself)

myself = np.reshape(myself, (1,13))

myselfDF = pd.DataFrame(myself)

myselfDF.columns = ['Age', 'Weight (lbs)', 'Height (inches)', 'Neck circumference (cm)', 'Chest circumference (cm)', 

'Abdomen 2 circumference (cm)', 'Hip circumference (cm)', 'Thigh circumference (cm)', 'Knee circumference (cm)',

'Ankle circumference (cm)', 'Biceps (extended) circumference (cm)', 'Forearm circumference (cm)', 'Wrist circumference (cm)']

myselfDF
y = df['Underwater weighing density']

y2 = df['Body fat from equation']

y_y = pd.concat([y,y2],axis=1)

X = df.drop(['Body fat from equation','Underwater weighing density'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

RF = RandomForestRegressor(n_estimators=99, random_state=42)

RF.fit(X_train, y_train)

print(sqrt(mean_squared_error(y_test, RF.predict(X_test))))

print(median_absolute_error(y_test, RF.predict(X_test)))
preds = np.stack([t.predict(X_test) for t in RF.estimators_])

preds[:,0], np.mean(preds[:,0]), y_test

plt.plot([r2_score(y_test, np.mean(preds[:i+1], axis=0)) for i in range(100)]);
# Predicting underwater weighing density

RF.predict(myselfDF)
X_train_all, X_test_all, yy_train, yy_test = train_test_split(X, y_y, test_size=0.2)

RF_multi = RandomForestRegressor(n_estimators=100, random_state=42)
RF_multi.fit(X_train_all, yy_train)

print(mean_squared_error(yy_test, RF_multi.predict(X_test_all)))

#print(median_absolute_error(yy_test, RF.predict(X_test_all)))
RF_multi.predict(myselfDF)