import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')
df.drop(['id','b10','b12','b26','b61','b81'],axis = 1 ,inplace = True)
from sklearn.ensemble import RandomForestRegressor
X = df.iloc[0:,0:97]

y = df.iloc[0:,97]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)
rf0 = RandomForestRegressor(max_depth=40,n_estimators=200,n_jobs = -1,min_samples_leaf = 1 , min_samples_split = 2 )
rf0.fit(X_train , y_train)
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test, rf0.predict(X_test)))
dft = pd.read_csv('test.csv')
dft.drop(['b10','b12','b26','b61','b81'],axis = 1 ,inplace = True)
sub0 = pd.DataFrame().append(dft['id']).astype('int32')
dft.drop(['id'],axis = 1 ,inplace = True)
y_pred = rf0.predict(dft)
y_pred = np.reshape(y_pred , (40292,1))
sub0 = np.transpose(sub0)
sub0.insert(1,'label' , y_pred , True)
sub0.head()
sub0.to_csv('final3.csv',index = False)