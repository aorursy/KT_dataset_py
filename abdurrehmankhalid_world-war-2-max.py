import pandas as pandasInstance
import numpy as numpyInstance
import matplotlib.pyplot as matplotlibInstance
import seaborn as seabornInstance
%matplotlib inline
worldWar2Data = pandasInstance.read_csv('../input/Summary of Weather.csv')
worldWar2Data.head()
worldWar2Data.info()
worldWar2Data.describe()
seabornInstance.scatterplot(x='MaxTemp',y='MinTemp',data=worldWar2Data)
matplotlibInstance.figure(figsize=(15,10))
matplotlibInstance.tight_layout()
seabornInstance.distplot(worldWar2Data['MaxTemp'])
matplotlibInstance.figure(figsize=(15,10))
matplotlibInstance.tight_layout()
seabornInstance.heatmap(worldWar2Data.corr(),cmap='magma',annot=True)
worldWar2Data.columns
X_Features = worldWar2Data[['MinTemp']]
X_Features
Y_Predict = worldWar2Data['MaxTemp']
Y_Predict
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_Features, Y_Predict, test_size=0.40, random_state=101)
from sklearn.linear_model import LinearRegression
linearRegressionInstance = LinearRegression()
linearRegressionInstance.fit(X_train,y_train)
linearRegressionInstance.intercept_
linearRegressionInstance.coef_
coefficientDescription = pandasInstance.DataFrame(linearRegressionInstance.coef_,index=X_train.columns,columns=['Co-Efficient Values'])
coefficientDescription
maxTempPredictions = linearRegressionInstance.predict(X_test)
maxTempPredictions
matplotlibInstance.scatter(y_test,maxTempPredictions)
matplotlibInstance.figure(figsize=(15,10))
matplotlibInstance.tight_layout()
seabornInstance.distplot(y_test - maxTempPredictions)
from sklearn import metrics
metrics.mean_absolute_error(y_test,maxTempPredictions)
metrics.mean_squared_error(y_test,maxTempPredictions)
numpyInstance.sqrt(metrics.mean_squared_error(y_test,maxTempPredictions))
coefficientDescription
