import numpy as np
import pandas as pd
train = pd.read_csv("../input/train_V2.csv")
train.head()
train2 = pd.concat([train, pd.get_dummies(train['matchType'])], axis=1)
train2.drop(['Id', 'groupId', 'matchId', 'matchType'], axis=1, inplace=True)
train2.head()
train2.fillna(0)
train2 = train2.dropna(axis=0)
X = train2.drop('winPlacePerc', axis=1)
y = train2['winPlacePerc']
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
missing_val_count_by_column = (train2.isnull().sum())
print(missing_val_count_by_column)
reg.fit(X, y)
from sklearn.model_selection import cross_val_score
cross_val_score(LinearRegression(), X, y, scoring='neg_mean_absolute_error').mean()
cols_to_fit = [col for col in X.columns]
corr = train2[cols_to_fit].corr()
corr[cols_to_fit[0]]
newFeatures = []
for i in cols_to_fit:
    for j in cols_to_fit:
        if(i==j):
            continue
        newFeatures.append([corr[i][j], i, j])
        print(i+' '+j)
newFeatures = sorted(newFeatures)
newFeatures
for i in range(50):
    print(i)
    X[newFeatures[i][1]+newFeatures[i][2]] = X[newFeatures[i][1]]*X[newFeatures[i][2]]
cross_val_score(LinearRegression(), X, y, scoring='neg_mean_absolute_error').mean()
