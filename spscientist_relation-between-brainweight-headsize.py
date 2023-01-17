import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = pd.read_csv("../input/headbrain.csv")
data.head()
data.describe()
sns.regplot(x = data['Head Size(cm^3)'], y = data['Brain Weight(grams)']);
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values
X = X.reshape((len(X),1))
reg = LinearRegression()
reg = reg.fit(X,Y)
Y_pred = reg.predict(X)
rmse = np.sqrt(mean_squared_error(Y,Y_pred))
print("RMSE =", rmse)
r2 = reg.score(X,Y)
print("R2 Score =", r2)