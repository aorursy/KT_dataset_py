import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.model_selection import train_test_split



"""

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots

"""
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
x_test = test[input_cols]
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X_scaled = scaler.transform(X)

x_test_scaled = scaler.transform(x_test)

"""
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

pca.fit(X)

X_pca_scaled = pca.transform(X)

x_test_pca_scaled = pca.transform(x_test)
X_trainval, X_test, y1_trainval, y1_test, y2_trainval, y2_test = train_test_split(X_pca_scaled, Y1, Y2, random_state = 0)

X_train, X_valid, y1_train, y1_valid, y2_train, y2_valid = train_test_split(X_trainval, y1_trainval, y2_trainval, random_state = 1)

print("훈련 세트의 크기: {},  검증 세트의 크기: {},  테스트 세트의 크기: {}".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



kfold = KFold(n_splits=5, shuffle = True, random_state = 0)



best_score = 0

for max_features in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:

    rf = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', max_features = max_features)

    scores = cross_val_score(rf, X_trainval, y1_trainval, cv=kfold)

    score = np.mean(scores)

    if score > best_score:

        best_score = score

        best_parameters = {'max_features': max_features}

rf = RandomForestClassifier(**best_parameters, criterion = 'entropy')

rf.fit(X_trainval, y1_trainval)

test_score = rf.score(X_test, y1_test)

print("검증 세트에서 최고 점수: {:.2f}".format(best_score))

print("최적 매개변수: ", best_parameters)

print("최적 매개변수에서 테스트 세트 점수: {:.2f}".format(test_score))
rf = RandomForestClassifier(**best_parameters, criterion = 'entropy')

rf.fit(X_trainval, y1_trainval)
print("훈련 세트 정확도: {:.3f}".format(rf.score(X_trainval, y1_trainval)))

print("테스트 세트 정확도: {:.3f}".format(rf.score(X_test, y1_test)))
pred1 = rf.predict(x_test_pca_scaled)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



kfold = KFold(n_splits=5, shuffle = True, random_state = 0)



best_score = 0

for max_features in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:

    rf = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', max_features = max_features)

    scores = cross_val_score(rf, X_trainval, y2_trainval, cv=kfold)

    score = np.mean(scores)

    if score > best_score:

        best_score = score

        best_parameters = {'max_features': max_features}

rf = RandomForestClassifier(**best_parameters, criterion = 'entropy')

rf.fit(X_trainval, y2_trainval)

test_score = rf.score(X_test, y2_test)

print("검증 세트에서 최고 점수: {:.2f}".format(best_score))

print("최적 매개변수: ", best_parameters)

print("최적 매개변수에서 테스트 세트 점수: {:.2f}".format(test_score))
rf = RandomForestClassifier(**best_parameters, criterion = 'entropy')

rf.fit(X_trainval, y2_trainval)
print("훈련 세트 정확도: {:.3f}".format(rf.score(X_trainval, y2_trainval)))

print("테스트 세트 정확도: {:.3f}".format(rf.score(X_test, y2_test)))
pred2 = rf.predict(x_test_pca_scaled)
ids.shape
pred1.shape
pred2.shape
output = pd.DataFrame({ 'ForecastId' : ids, 'ConfirmedCases': pred1,'Fatalities':pred2 })

output

output.to_csv('submission.csv', index=False)