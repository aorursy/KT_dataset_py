import statsmodels.api as sm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sklearn
import sklearn.linear_model
import matplotlib
import sklearn.metrics

data = pd.read_csv('/kaggle/input/cricket-chirp-vs-temperature/Cricket_chirps.csv')
data.head()
data.dtypes
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html
data = data.rename(columns={"X": "Chirps_Per_Sec", "Y": "Temperature"})
data.head()
data.describe()
sns.kdeplot(data['Chirps_Per_Sec'], shade=True)
sns.kdeplot(data['Temperature'], shade=True)
data.plot.scatter(x = "Temperature", y = "Chirps_Per_Sec")
matplotlib.pyplot.boxplot(data["Chirps_Per_Sec"])
matplotlib.pyplot.boxplot(data["Temperature"])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data.loc[:,["Temperature"]], data["Chirps_Per_Sec"], test_size=0.33, random_state=42)
y_train.head()
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
linear_regression_model = sklearn.linear_model.LinearRegression(fit_intercept = True)

linear_regression_model.fit(X_train,y_train)
linear_regression_model.coef_
linear_regression_model.intercept_
y_pred = linear_regression_model.predict(X_test)
linear_regression_model.score(X_test, y_test)
matplotlib.pyplot.scatter(X_train, y_train, color = 'green')
matplotlib.pyplot.scatter(X_test, y_test, color = 'blue')   
matplotlib.pyplot.scatter(X_test, y_pred, color = 'blue')  # The predicted temperatures of the same X_test input.
matplotlib.pyplot.plot(X_train, linear_regression_model.predict(X_train), color = 'gray')
matplotlib.pyplot.title('Temperature based on chirp count')
matplotlib.pyplot.xlabel('Chirps/minute')
matplotlib.pyplot.ylabel('Temperature')
matplotlib.pyplot.show()
sklearn.metrics.mean_absolute_error(y_test, y_pred)
sklearn.metrics.mean_squared_error(y_test, y_pred) 
sklearn.metrics.r2_score(y_test, y_pred, multioutput='variance_weighted')
l_r_model = sm.OLS(y_train,sm.add_constant(X_train))

model_results = l_r_model.fit()

model_results.summary()
