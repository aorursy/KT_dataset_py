import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.get('DESCR'))
df_x = pd.DataFrame(boston.get('data'),columns=boston.get('feature_names'))
df_y = pd.DataFrame(boston.get('target'))
df_y = np.array(df_y)
df_x.head()

X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(df_x, df_y, test_size = 0.33, random_state = 5)
reg = linear_model.LinearRegression()
reg.fit(X_train,Y_train)
Y_predict = reg.predict(X_test)
plt.scatter(Y_test, Y_predict, color='blue')
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()
error = np.mean((Y_predict - Y_test)**2)
print('Mean Squared Error: ' + str(error))
error = mean_squared_error(Y_test, Y_predict)
print('Mean Squared Error: ' + str(error))
reg.fit(df_x,df_y)
Y_predict2 = reg.predict(df_x)
plt.scatter(df_y, Y_predict2, color='green')
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices (More data)")
plt.show()
error = np.mean((Y_predict2 - df_y)**2)
print('Mean Squared Error when model is trained using entire dataset: ' + str(error))