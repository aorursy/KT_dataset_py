import numpy
import matplotlib.pyplot as plt
import pandas
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
numpy.random.seed(7)
dataframe = pandas.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv', engine='python')
dataframe.head()
dataframe = dataframe.drop(['County','Province_State','Country_Region','Target'], axis=1)
dataframe.head()
dataframeTest = pandas.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv', engine='python')
dataframeTest.head()
dataframeTest = dataframeTest.drop(['County','Province_State','Country_Region','Target'], axis=1)
dataframeTest.head()
dataframe['Date']=pandas.to_datetime(dataframe['Date'])
dataframeTest['Date']=pandas.to_datetime(dataframeTest['Date'])
dataframeTest['Date']=dataframeTest['Date'].dt.strftime("%Y%m%d")
dataframe['Date']=dataframe['Date'].dt.strftime("%Y%m%d")
dataframe.head()
dataframeTest.head()
X = dataframe.drop(['TargetValue', 'Id'], axis=1)
X.head()
Y = dataframe['TargetValue']
Y.head()
X_train, X_validate, y_train, y_validate = train_test_split(X, Y, test_size = 0.20, random_state = 0)
X_train.shape, X_validate.shape, y_train.shape, y_validate.shape
X_train = X_train.to_numpy()
print(X_train)
X_train.shape
X_validate = X_validate.to_numpy()
print(X_validate)
X_validate.shape
y_validate = y_validate.to_numpy()
print(y_validate)
y_validate.shape
y_train = y_train.to_numpy()
print(y_train)
y_train.shape
model1 = MLPRegressor(hidden_layer_sizes=(100, 50, 25 ,12, 6, 3, ), activation='relu', solver='adam', alpha=0.0001, max_iter=100)

scores = []

pipeline = Pipeline([('scaler2' , StandardScaler()),
                        ('MLPRegressor: ', model1)])
pipeline.fit(X_train , y_train)
prediction = pipeline.predict(X_validate)
print(prediction)
scores.append(pipeline.score(X_validate, y_validate))
scores
len(prediction), len(y_validate)
mean_squared_error(y_validate, prediction)
ids = dataframeTest['ForecastId']
ids
ids = ids.to_numpy()
ids
X_test = dataframeTest.drop(['ForecastId'], axis=1)
X_test.head()
X_test = X_test.to_numpy()
print(X_test)
X_test.shape
y_test_pred = pipeline.predict(X_test)
print(len(y_test_pred))
y_test_pred
output = pandas.DataFrame({'Id': ids, 'TargetValue': y_test_pred})
output.head()
q005=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
q05=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
q095=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
q005.columns=['Id','q0.05']
q05.columns=['Id','q0.5']
q095.columns=['Id','q0.95']
submission=pandas.concat([q005,q05['q0.5'],q095['q0.95']],1)
submission.head()
submission['q0.05']=submission['q0.05'].clip(0,10000)
submission['q0.5']=submission['q0.5'].clip(0,10000)
submission['q0.95']=submission['q0.95'].clip(0,10000)
submission
submission=pandas.melt(submission, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
submission
submission['variable']=submission['variable'].str.replace("q","", regex=False)
submission
submission['ForecastId_Quantile']=submission['Id'].astype(str)+'_'+submission['variable']
submission
submission['TargetValue']=submission['value']
submission=submission[['ForecastId_Quantile','TargetValue']]
submission.reset_index(drop=True,inplace=True)
submission.head()
submission.to_csv("submission.csv",index=False)
len(X_test)*3