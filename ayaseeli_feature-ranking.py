import numpy as np

import pandas as pd 

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
house = pd.read_csv("../input/kc_house_data.csv")

del house["id"]

del house["date"]
X = house[house.columns[1:19]]

Y = house["price"]

colnames = X.columns
ranks = {}

# Create our function which stores the feature rankings to the ranks dictionary

def ranking(ranks, names, order=1):

    minmax = MinMaxScaler()

    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]

    ranks = map(lambda x: round(x,2), ranks)

    return dict(zip(names, ranks))
lr = LinearRegression(normalize=True)

lr.fit(X,Y)

rfe = RFE(lr, n_features_to_select=1, verbose =3 )

rfe.fit(X,Y)

ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), colnames)
rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)

rf.fit(X,Y)

ranks["RF"] = ranking(rf.feature_importances_, colnames)
ranks