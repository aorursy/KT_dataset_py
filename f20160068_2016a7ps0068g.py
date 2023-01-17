import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("train.csv")

df.fillna(value=df.mean(),inplace=True)

df = pd.get_dummies(data=df,columns=['type'])

out = pd.read_csv("test.csv")

out.fillna(value=out.mean(),inplace=True)

out = pd.get_dummies(data=out,columns=['type'])

numerical_features_sol1 = ['id','feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11','type_new']

numerical_features_sol2 = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11','type_new']

X = df[numerical_features_sol1]

y = df["rating"]
from sklearn.ensemble import ExtraTreesRegressor



clf = ExtraTreesRegressor(n_estimators=900,max_depth=25,random_state=69).fit(X,y)

y_pred = [round(y) for y in clf.predict(out[numerical_features_sol1])]



upload = pd.concat([out.id,pd.DataFrame(data=y_pred)],axis=1)

upload.columns = ['id','rating']

upload.to_csv('submit.csv')