import numpy as np
import pandas as pd
df_test = pd.read_csv("../input/test.csv")
df_train = pd.read_csv("../input/train.csv")

df_train = df_train[df_train["GarageArea"] < 1200]
data = df_train.select_dtypes(include=[np.number]).interpolate().dropna()
y = np.log(df_train.SalePrice)
x = data.drop(["SalePrice", "Id"], axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.33)
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
from sklearn import svm
clf = svm.SVR(kernel='rbf', C=1.0)
from sklearn.model_selection import GridSearchCV
# clf = GridSearchCV(svm.SVR(kernel='rbf'), cv=5, param_grid={'C':[1e0, 1e1, 1e2, 1e3]})
clf.fit(X_train, y_train)
min_max_scaler_test = preprocessing.MinMaxScaler()
X_test = min_max_scaler_test.fit_transform(X_test)
y_predict = clf.predict(X_test)
from sklearn.metrics import mean_squared_error
print (mean_squared_error(y_test, y_predict))