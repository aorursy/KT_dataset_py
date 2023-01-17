import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv('../input/parkinsons-data-set/parkinsons.data')
df.head()
df.describe()
df.info()
df.isnull().any()
df.shape
df['status'].value_counts()
import seaborn as sns

sns.countplot(df['status'])
df.dtypes
X = df.drop(['name'], 1)

X = np.array(X.drop(['status'], 1))

y = np.array(df['status'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))
X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
from xgboost import XGBClassifier

model = XGBClassifier().fit(X_train, y_train)
predictions = model.predict(X_test)

predictions
y_test
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))