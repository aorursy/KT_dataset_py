import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


import pandas as pd
df = pd.read_csv("../input/indonesia-coronavirus-cases/jabar.csv")
#menggunakan data Jabar
df.columns = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39']
print(df.describe())
X = df.iloc[:,1:39]
Y = df.iloc[:, 8]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['jumlah'])
print(coeff_df)
y_pred = model.predict(X_test)
df = pd.DataFrame({'sebenarnya': Y_test, 'prediksi': y_pred})
print(df.head(10))
df.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='1', color='green')
plt.grid(which='minor', linestyle=':', linewidth='1', color='black')
plt.show()
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))      
r2_value = r2_score(Y_test, y_pred)                     

print("Intercept: \n", model.intercept_)
print("Root Mean Square Error \n", rmse)
print("R^2 Value: \n", r2_value)