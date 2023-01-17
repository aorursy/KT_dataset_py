import pandas as pd

avgmdata= pd.read_csv("../input/Ag_data.csv")
avgmdata
X = avgmdata[['windgust','pressure','temperature','winddegree','windspeed','t1','t2','t3','t4','t5',"t6","t7","t8",'t9','t10']]

y = avgmdata[['AG']]
X
y
X.isna().sum()




X.fillna(0,inplace=True)

y.fillna(0,inplace=True)



X.isna().sum()
from sklearn import ensemble

from sklearn import datasets

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_error

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X, y)
t=[[26,1014,22,341,12,80.91,80.13,79.46,79.23,78.58,77.04,75.56,76.45,78.74,81.95]]

clf.predict(t)
import pickle

pickle.dump(clf,open("AG_XGB.sav",'wb'))