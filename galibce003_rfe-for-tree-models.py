import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
df = pd.read_csv('../input/cardiotocographic/Cardiotocographic.csv')
df.head()
df.shape
df.isnull().sum()
X = pd.DataFrame(df.iloc[:, 0:21])
y = df['NSP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)
rf = RandomForestClassifier()
model = rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf.score(X_test, y_test)
print(rf.feature_importances_)
rfe = RFE(estimator = RandomForestClassifier(), n_features_to_select = 18, step = 10, verbose = 1)
rfe.fit(X_train, y_train)
y1_pred = rfe.predict(X_test)
rfe.score(X_test, y_test)
print(X.columns[rfe.support_])