import pandas as pd
data = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
data = pd.get_dummies(data)
corr = data.corr()
corr["class_e"].sort_values(ascending=False)
from sklearn.metrics import accuracy_score
y_true = data["class_e"]

y_pred = data["odor_n"]
accuracy_score(y_true, y_pred)