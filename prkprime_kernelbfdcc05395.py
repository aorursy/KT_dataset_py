import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
model = pd.read_csv('../input/train_V2.csv')
model = model.dropna(axis='rows')
x = model.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]
y = model.iloc[:, 28]
model = RandomForestRegressor(n_estimators=20,random_state=0)
model.fit(x,y)
test = pd.read_csv('../input/test_V2.csv')
test = test.dropna(axis='rows')
x = test.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]
y = model.predict(x)
x = test.iloc[:, 0]
out = np.vstack((x, y))
out = np.transpose(out)
out = pd.DataFrame(out, columns=['Id', 'winPlacePerc'])
out.to_csv('submission1.csv', index=False)