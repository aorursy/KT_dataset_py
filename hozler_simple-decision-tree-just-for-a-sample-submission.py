import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train.head()
df_test.head()
print("length of train dataset:", len(df_train))
print("length of test dataset:", len(df_test))
y = df_train["target"].values
X = df_train[["card_id", "feature_1", "feature_2", "feature_3"]].values
Xsub = df_test[["card_id", "feature_1", "feature_2", "feature_3"]].values

print("input features for train set:\n",X)
print("--------------------")
print("input features for test set:\n",Xsub)
print("--------------------")
print("output:\n",y)
model = DecisionTreeRegressor(max_depth=5)
model.fit(X[:,1:], y)
y_pred = model.predict(Xsub[:,1:])
y_pred = y_pred.reshape(len(y_pred),1)
resultarray = np.append(Xsub, y_pred, axis=1)
print(resultarray)
resultdf = pd.DataFrame(resultarray, columns=["card_id", "f1", "f2", "f3", "target"])
resultdf = resultdf.drop(['f1', 'f2', 'f3'], axis=1)
resultdf.head()
resultdf.to_csv("submission.csv", sep=',', index=False)