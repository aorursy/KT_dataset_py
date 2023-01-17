import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
data = pd.read_csv("/kaggle/input/machine-learning-lab-cas-data-science-hs-20/train-data.csv", index_col=0)
X_data = data.drop(columns='G3')[['G1', 'G2']]
y_data = data['G3'].to_numpy()
X_train, X_dev, y_train, y_dev = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_dev_pred = model.predict(X_dev)
print('MAE\t', mean_absolute_error(y_dev, y_dev_pred))
# Baseline just guess the average

y_dev_pred_base = (X_dev['G1'].to_numpy() + X_dev['G2'].to_numpy()) / 2

print('MAE\t', mean_absolute_error(y_dev, y_dev_pred_base))