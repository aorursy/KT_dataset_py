import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
data.info()
data.head(150)
data.head()
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
sns.pairplot(data, hue="species")
print(data.iloc[:,4])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(dtype='int'), [4])], remainder='passthrough')
#y = np.array(ct.fit_transform(y))
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
np.set_printoptions(precision=2)
con = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), axis=1)
print(con)
from sklearn.metrics import mean_squared_error
err = mean_squared_error(y_test,y_pred)
print(err)