import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
pd.set_option('display.max_columns', 500)
df_train = pd.read_csv('../input/train_V2.csv')
df_test = pd.read_csv('../input/test_V2.csv')
df_train.head()
df_test.head()
df_train.columns[df_train.isna().any()].tolist()
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc2 = LabelEncoder()
matchType = df_train['matchType']
matchTypeTest = df_train['matchType']
enc.fit(matchType)
enc2.fit(matchTypeTest)
enc.classes_
matchTypes  = enc.transform(matchType)
matchTypes2 = enc.transform(matchTypeTest)
dataframesMatchType = pd.DataFrame(matchTypes)
dataframesMatchType2 = pd.DataFrame(matchTypes2)
df_train.drop('matchType', axis=1, inplace=True)
df_test.drop('matchType', axis=1, inplace=True)
df_train.head()
df_train['matchType'] = dataframesMatchType[0]
df_test['matchType'] = dataframesMatchType2[0]
df_train.head()
df_train.dropna(inplace=True)
df_train.columns[df_train.isna().any()].tolist()
plt.figure(figsize = (25,25))
sns.heatmap(df_train.corr(), annot=True)
df_train.describe()
from sklearn.model_selection import train_test_split
X = df_train.drop(['winPlacePerc', 'Id', 'groupId', 'matchId'], axis=1)
y = df_train['winPlacePerc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.30)
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.neural_network import MLPRegressor
bayes = MLPRegressor()
bayes.fit(X_train2, y_train2)
predictions = bayes.predict(X_test2)
from sklearn.metrics import mean_absolute_error, classification_report, mean_squared_error
predictions1 = bayes.predict(X_test)
print(mean_absolute_error(y_test, predictions1))
print(mean_absolute_error(y_test2, predictions))
myPredictions = df_test.drop(['Id', 'groupId', 'matchId'], axis=1)
predictions2 = bayes.predict(myPredictions)
predictions2
df_test['winPlacePerc'] = predictions2
dfFinal = df_test[['Id', 'winPlacePerc']]
dfFinal.head()
dfFinal.to_csv('pubgKaggle.csv', index=False)
