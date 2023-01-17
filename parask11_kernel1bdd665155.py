import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')
ids = submission['ForecastId']
input_cols = ["Lat","Long","Date"]

output_cols = ["ConfirmedCases","Fatalities"]
for i in range(df.shape[0]):

    df["Date"][i] = df["Date"][i][:4] + df["Date"][i][5:7] + df["Date"][i][8:]

    df["Date"][i] = int(df["Date"][i])
for i in range(test.shape[0]):

    test["Date"][i] = test["Date"][i][:4] + test["Date"][i][5:7] + test["Date"][i][8:]

    test["Date"][i] = int(test["Date"][i])
X = df[input_cols]

Y1 = df[output_cols[0]]

Y2 = df[output_cols[1]]
X_test = test[input_cols]
sk_tree = DecisionTreeClassifier(criterion='entropy')
sk_tree.fit(X,Y1)
pred1 = sk_tree.predict(X_test)
sk_tree.fit(X,Y2)
pred2 = sk_tree.predict(X_test)
ids.shape
pred1.shape
pred2.shape
output = pd.DataFrame({ 'ForecastId' : ids, 'ConfirmedCases': pred1,'Fatalities':pred2 })

output.to_csv('submission.csv', index=False)