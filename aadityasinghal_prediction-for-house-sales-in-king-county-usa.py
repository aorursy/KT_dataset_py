import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
dataset = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
dataset.head()
dataset.isnull().sum()
sns.heatmap(dataset.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
dataset.info()
X = dataset.iloc[:, [3,4,5,6,7,8,9,10,11,12,13,17,18,19,20]].values
y = dataset.iloc[:, 2].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
regressor = RandomForestRegressor(n_estimators = 30)
regressor.fit(X_train, y_train)
y_pred_train = regressor.predict(X_train)
print(r2_score(y_train,y_pred_train))
y_pred_test = regressor.predict(X_test)
print(r2_score(y_test,y_pred_test))