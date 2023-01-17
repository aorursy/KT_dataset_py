
import pandas as pd
import matplotlib.pyplot as plt
import fbprophet
test = pd.read_csv('../input/predice-el-futuro/test_csv.csv')
train = pd.read_csv('../input/predice-el-futuro/train_csv.csv')
train.head()
train.info()
train_copy = train.copy()
train = train.drop(columns='id')
train.head()
train['time'] = pd.to_datetime(train['time'])
train.columns = ['ds','y']
train.head()
train.tail()
from fbprophet import Prophet
model = Prophet()
model.fit(train)
test.info()
test['time'] = pd.to_datetime(test['time'])
test = test.drop(columns='id')
test.head()
test.columns = ['ds']
prediction=model.predict(test)
model.plot(prediction)
model.plot_components(prediction)
prediction.head()
prediction_copy = prediction[['ds','trend']]
prediction_copy = prediction_copy.drop(columns='ds')
prediction_copy.head()
prediction.head()
prediction_copy.to_excel("Solution.xlsx")