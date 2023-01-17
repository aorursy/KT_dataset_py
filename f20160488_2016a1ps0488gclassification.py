import pandas as pd

import numpy as np

from xgboost import XGBClassifier

import seaborn as sns

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv("../input/train_lab3.csv")

test = pd.read_csv("../input/test_data.csv")

train.head()
train_new = train.drop(['Unnamed: 0', 'soldierId'], axis=1)

test_new = test.drop(['Unnamed: 0', 'soldierId'], axis=1)

test_new.describe()
drop_columns = ['shipId', 'attackId', 'throatSlits', 'killingStreaks', 'horseRideKills', 'friendlyKills', 'castleTowerDestroys']

train_new = train_new.drop(drop_columns, axis=1)

test_new = test_new.drop(drop_columns, axis=1)
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train_new.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
y = train_new['bestSoldierPerc'].ravel()

train_new = train_new.drop(['bestSoldierPerc'], axis=1)

X = train_new.values # Creates an array of the train data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
X_train = X

y_train = y

X_test = test_new.values
model = XGBClassifier(n_estimators=1000, max_depth=3)

model.fit(X_train, y_train)
print(model.feature_importances_, list(train_new.columns))
from sklearn import metrics

y_pred = model.predict(X_test)

# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(y_pred)
answer = {"soldierId" : test["soldierId"], "bestSoldierPerc" : y_pred}

ans = pd.DataFrame(answer, columns = ["soldierId", "bestSoldierPerc"])

ans.to_csv("answer.csv", index = False)