import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (15,7)
#plt.rcParams['font.cursive'] = ['Source Han Sans TW', 'sans-serif']
from sklearn.model_selection import train_test_split
housing_data_set = pd.read_csv("../input/kc_house_data.csv")
housing_data_set.head()
housing_data_set.describe()
housing_data_set.shape
#plt.figure(figsize=(17,6))
sns.heatmap(housing_data_set.corr(),cmap='viridis',annot=True)
housing_data_set.info()
# Looking for nulls
print(housing_data_set.isnull().any())
# Inspecting type
print(housing_data_set.dtypes)
# Dropping the id and date columns
house = housing_data_set.drop(['id', 'date'],axis=1)
y = house['price']
X = house.drop(['price'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.coef_
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['coeff'])
cdf.head()
#Predictions
predication = lm.predict(X_test)
predication
plt.scatter(y_test,predication)
#Residual
sns.distplot((y_test-predication))
from sklearn import metrics
metrics.mean_absolute_error(y_test,predication)
metrics.mean_squared_error(y_test,predication)
np.sqrt(metrics.mean_squared_error(y_test,predication))
#Building the optimal model using Backwards Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((21613,1)).astype(int), values= X,axis=1)
X_opt = house.drop(['price'], axis=1)
lm_OLS = sm.OLS(endog= y,exog= X_opt).fit()
lm_OLS.summary()


