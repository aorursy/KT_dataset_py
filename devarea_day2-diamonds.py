import seaborn as sb

import pandas as pd
df = sb.load_dataset('diamonds')
df.info()
df.head()
df.cut.unique()
df.describe()
df.corr()
df.drop(['cut','color','clarity'],axis=1,inplace=True)
df = pd.get_dummies(df , columns=['cut','color','clarity'],drop_first=True)
df.head(20)
df.info()
X = df.drop(['price','table','depth'],axis=1)
y = df.price
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33)
import sklearn.linear_model as sl
model = sl.LinearRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)
model.score(X_test,y_test)
model.score(X_train,y_train)
y_test[(y_test - pred) < 10]
sb.distplot(y_test - pred)
sb.scatterplot(y_test , pred)
import sklearn.metrics as mt
mt.mean_absolute_error(y_test,pred)
import numpy as np
np.sqrt(mt.mean_squared_error(y_test,pred))
model.coef_
model.intercept_
pred.max()
sb.distplot(pred)