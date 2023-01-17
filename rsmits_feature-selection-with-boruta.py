import numpy as np 

import pandas as pd 

from sklearn.ensemble import RandomForestClassifier

from boruta import BorutaPy
train = pd.read_csv("../input/application_train.csv")

train.shape
train = pd.get_dummies(train, drop_first=True, dummy_na=True)

train.shape
features = [f for f in train.columns if f not in ['TARGET','SK_ID_CURR']]

len(features)
train[features] = train[features].fillna(train[features].mean()).clip(-1e9,1e9)
X = train[features].values

Y = train['TARGET'].values.ravel()
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
boruta_feature_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=4242, max_iter = 50, perc = 90)

boruta_feature_selector.fit(X, Y)
X_filtered = boruta_feature_selector.transform(X)

X_filtered.shape
final_features = list()

indexes = np.where(boruta_feature_selector.support_ == True)

for x in np.nditer(indexes):

    final_features.append(features[x])

print(final_features)