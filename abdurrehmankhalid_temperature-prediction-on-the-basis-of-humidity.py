import numpy as numpyInstace
import pandas as pandasInstance
import matplotlib.pyplot as matplotlibInstace
import seaborn as seabornInstace
%matplotlib inline
weatherData = pandasInstance.read_csv('../input/weatherHistory.csv')
weatherData.head()
weatherData.info()
weatherData.describe()
seabornInstace.scatterplot(x='Temperature (C)',y='Humidity',data=weatherData)
seabornInstace.jointplot(x='Temperature (C)',y='Humidity',data=weatherData,kind='hex')
seabornInstace.lmplot(x='Temperature (C)',y='Humidity',data=weatherData)
matplotlibInstace.figure(figsize=(15,10))
matplotlibInstace.tight_layout()
seabornInstace.distplot(weatherData['Temperature (C)'])
matplotlibInstace.figure(figsize=(15,10))
matplotlibInstace.tight_layout()
seabornInstace.heatmap(weatherData.corr(),annot=True,cmap = 'coolwarm')
X_Features = weatherData[['Humidity']]
X_Features
Y_Prediction = weatherData[['Temperature (C)']]
Y_Prediction
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_Features, Y_Prediction, test_size=0.40, random_state=101)
from sklearn.linear_model import LinearRegression
linearRegressionInstance = LinearRegression()
linearRegressionInstance
linearRegressionInstance.fit(X_train,y_train)
linearRegressionInstance.intercept_
linearRegressionInstance.coef_
coefficientInformation = pandasInstance.DataFrame(data=linearRegressionInstance.coef_,index=X_train.columns,columns=['Co-Efficient Values'])
coefficientInformation
temperaturePredictions = linearRegressionInstance.predict(X_test)
temperaturePredictions
matplotlibInstace.scatter(y_test,temperaturePredictions)
seabornInstace.distplot(y_test-temperaturePredictions)
from sklearn import metrics
metrics.mean_absolute_error(y_test,temperaturePredictions)
metrics.mean_squared_error(y_test,temperaturePredictions)
numpyInstace.sqrt(metrics.mean_squared_error(y_test,temperaturePredictions))
coefficientInformation
