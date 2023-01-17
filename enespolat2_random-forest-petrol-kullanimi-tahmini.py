import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import init_notebook_mode, iplot, plot
import seaborn as sns
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    
data = pd.read_csv("/kaggle/input/petrol-consumption/petrol_consumption.csv",encoding='ISO-8859-1')
data.head()
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rf = RandomForestRegressor(n_estimators = 100 , random_state = 42)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
MSE_total =[]
Tree_Sayisi=[]

for m in range(1,201,10):
    Tree_Sayisi.append(m)
    
for i in range(1,201,10):
    rf = RandomForestRegressor(n_estimators = i , random_state = 42)
    rf.fit(X_train,y_train)
    y_pred_person = rf.predict(X_test)
    MSE_person = np.sqrt(metrics.mean_squared_error(y_test, y_pred_person))
    MSE_total.append(MSE_person)
    
    
MSE_liste = pd.DataFrame({"Tree Sayısı" : Tree_Sayisi , "MSE Oranları" : MSE_total})

MSE_liste
fig = go.Figure()
fig.add_trace(go.Scatter(x=MSE_liste["Tree Sayısı"], y=MSE_liste['MSE Oranları'],
                    mode='lines+markers',
                    name='lines+markers'))
fig.show()
