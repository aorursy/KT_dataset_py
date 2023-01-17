import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))

df = pd.read_csv('../input/Breast_cancer_data.csv')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

y = df.diagnosis
X = df.drop(['diagnosis'],axis=1)
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25, random_state = 42)
import xgboost as xgb

model = xgb.XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.005).fit(train_X, train_y)
predictions = model.predict(test_X)
from sklearn.metrics import accuracy_score
print("Acurracy : " + str(100*accuracy_score(predictions, test_y)))
