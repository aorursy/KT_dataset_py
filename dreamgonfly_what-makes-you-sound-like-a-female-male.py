import pandas as pd
xy = pd.read_csv('../input/voice.csv')



X = xy.drop('label', axis='columns')

y = xy['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, rf.predict(X_test))