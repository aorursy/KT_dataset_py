# Directive pour afficher les graphiques dans Jupyter
%matplotlib inline
# Pandas : librairie de manipulation de données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import model_selection

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/hourly-energy-consumption/AEP_hourly.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values(by = 'Datetime')
df.head()
plt.figure(figsize=(20,10))
plt.plot(df['Datetime'] , df['AEP_MW'])
from fbprophet import Prophet
df_prophet = df.copy()
df_prophet.columns = ['ds','y']
df_prophet.head()
train = df_prophet[df_prophet['ds'] < '2018-01-01'].copy()
test = df_prophet[df_prophet['ds'] >= '2018-01-01'].copy()
plt.figure(figsize=(20,10))
plt.plot(train['ds'] , train['y'], color = 'blue')
plt.plot(test['ds'] , test['y'], color = 'grey')
plt.show()
prophet = Prophet()
prophet.fit(train)
prediction = prophet.predict(test)
prediction.head()
plt.figure(figsize=(20,10))
plt.plot(train['ds'] , train['y'], color = 'blue')
plt.plot(test['ds'] , test['y'], color = 'grey', alpha = 0.5)
plt.plot(prediction['ds'] , prediction['yhat'], color = 'orange', alpha = 0.5)
plt.show()
plt.figure(figsize=(20,10))
plt.plot(train['ds'] , train['y'], color = 'blue')
plt.plot(test['ds'] , test['y'], color = 'grey', alpha = 0.5)
plt.plot(prediction['ds'] , prediction['yhat_upper'], color = 'red', alpha = 0.5)
plt.plot(prediction['ds'] , prediction['yhat_lower'], color = 'green', alpha = 0.5)
plt.show()

