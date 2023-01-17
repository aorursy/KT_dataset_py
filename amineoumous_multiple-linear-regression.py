import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

dataset = pd.read_csv("../input/50-startups-data/50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

st = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = st.fit_transform(X)
X
X = X[:,1:]
X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('- y_pred : ')
print(y_pred)
print('- y_test : ')
print(y_test)
import numpy as np
X = np.append(arr = np.ones((50, 1)).astype(float), values = X, axis = 1)
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
import statsmodels.api as sm
model = sm.OLS(endog = y, exog = X_opt)
regressor_OLS = model.fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
X_opt = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = np.array(X[:, [0,3, 4, 5]], dtype=float)
#X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = np.array(X[:, [0, 3, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = np.array(X[:, [0,3]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()