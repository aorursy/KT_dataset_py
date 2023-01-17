import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/weatherAUS.csv')
df.head()
df.describe()
df.shape
df = df.dropna()
df.shape
plt.figure(figsize=(15, 10))
p = sns.heatmap(df.corr(), annot=True)
# dropping the features which are not useful for predictions
df = df.drop(['Location','Date','Evaporation','Sunshine', 'Cloud9am','Cloud3pm',
                           'WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am',
                           'WindSpeed3pm'], axis=1)
# creating the lables
Y = df["RainTomorrow"]
X = df.drop(["RainTomorrow"], axis=1)
X = X.replace({"No":0, "Yes":1})
X = X.fillna(0)
Y = Y.replace({"No":0, "Yes":1})
Y = Y.fillna(0)
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)
# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)
from sklearn.svm import SVC

clf = SVC(gamma='scale')
clf.fit(X_scaled, Y)
predictions = clf.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(predictions, Y_test)
accuracy
from sklearn.ensemble import RandomForestClassifier
clf_rfc = RandomForestClassifier(n_estimators=10)
clf_rfc.fit(X_train, Y_train)
predictions_rfc = clf_rfc.predict(X_test)
accuracy_rfc = accuracy_score(predictions_rfc, Y_test)
accuracy_rfc