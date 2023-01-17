import numpy as np

import pandas as pd 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt

from pandas import read_csv

from sklearn import tree

from sklearn import metrics
df = read_csv("../input/mushrooms.csv")
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

features = list(df.columns.values)

features.remove("class")

for i in features:

    df[i] = le.fit_transform(df[i])

df["class"] = df["class"].astype("category")

X = df[features]

Y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(X_train, y_train)

forest.score(X_train, y_train)
y_pred = forest.predict(X_test)

acc_test = metrics.accuracy_score(y_test,y_pred)

acc_test
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(forest.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

print (importances)

importances.plot.bar()