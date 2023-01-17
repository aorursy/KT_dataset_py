import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('../input/train.csv')
df.fillna(0, inplace=True)
df_test = pd.read_csv('../input/test.csv')
df_test.fillna(0, inplace=True)
df_test.head()
X_train = df.iloc[:, [2,5,6,7,9]].values
y_train = df.iloc[:, 1:2].values
X_test = df_test.iloc[:, [1,4,5,6,8]].values
from sklearn.preprocessing import StandardScaler
sclr = StandardScaler()
X_train = sclr.fit_transform(X_train)
X_test = sclr.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=3, criterion='entropy', n_estimators=10, min_samples_leaf=25,
                                    random_state=0, min_samples_split=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred
y_pred
PassengerId = df_test.iloc[:, [0]]
predicted = pd.DataFrame(y_pred)
predicted.columns = ['Survived']
result = pd.concat([PassengerId, predicted], axis=1)
result.to_csv('result.csv', index=False)



