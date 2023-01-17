import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from autofeat import AutoFeatRegression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import explained_variance_score
df = pd.read_csv("../input/Concrete_Data_Yeh.csv")
df.head()
X = df[['cement','slag','flyash','water','superplasticizer','coarseaggregate','fineaggregate','age']]

y = df[['csMPa']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = AutoFeatRegression()

model
X_train_feature_creation = model.fit_transform(X_train.to_numpy(), y_train.to_numpy().flatten())
X_test_feature_creation = model.transform(X_test.to_numpy())
X_test_feature_creation.head()
X_train_feature_creation.head()
X_train_feature_creation.shape[1] - X_train.shape[1]
model_1 = LinearRegression().fit(X_train,y_train.to_numpy().flatten())

model_2 = LinearRegression().fit(X_train_feature_creation, y_train.to_numpy().flatten())
explained_variance_score(y_test, model_1.predict(X_test)), explained_variance_score(y_test, model_2.predict(X_test_feature_creation))