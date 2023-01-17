# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
import folium
import os

import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        

data = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
data.head()
data_today= data[data['Date']=='27/03/20']
data_new = data_today.groupby(['State/UnionTerritory','Cured','Deaths'])['TotalCases'].sum().reset_index().sort_values('TotalCases',ascending=False)
data_new
data_today.columns
data_population = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")
data_population.head()
data_population.columns
#merging of data_today and data_population
data_with_population = data_population.merge(data_new,left_on='State / Union Territory',right_on='State/UnionTerritory')
data_with_population = data_with_population.drop(labels=['State/UnionTerritory','Sno'],axis=1)
data_with_population.head()
data_with_population = data_with_population.sort_values('Population',ascending = False)
data_with_population             # by this we can understand currently there is no relation between population and number of cases
data_hospitalbeds = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")
data_hospitalbeds.head()
data_population_hospitalbeds = data_with_population.merge(data_hospitalbeds,left_on='State / Union Territory',right_on='State/UT')
data_population_hospitalbeds = data_population_hospitalbeds.drop(labels=['State/UT','Sno'],axis =1)
data_population_hospitalbeds.dtypes
data_population_hospitalbeds= data_population_hospitalbeds.drop(labels=['Unnamed: 12','Unnamed: 13'],axis=1)

data_population_hospitalbeds.head()
data_population_hospitalbeds['NumUrbanHospitals_NHP18'] = data_population_hospitalbeds['NumUrbanHospitals_NHP18'].astype('int')
data_population_hospitalbeds['NumRuralHospitals_NHP18'] = data_population_hospitalbeds['NumRuralHospitals_NHP18'].astype('int')
#data_population_hospitalbeds['NumSubDistrictHospitals_HMIS'] = data_population_hospitalbeds['NumSubDistrictHospitals_HMIS'].astype('int')
data_population_hospitalbeds['NumDistrictHospitals_HMIS'] = data_population_hospitalbeds['NumDistrictHospitals_HMIS'].astype('int')

data_population_hospitalbeds['TotalHospitals_inState']= data_population_hospitalbeds['NumUrbanHospitals_NHP18']+data_population_hospitalbeds['NumRuralHospitals_NHP18']+data_population_hospitalbeds['NumDistrictHospitals_HMIS']

data_population_hospitalbeds.head()
fig = px.bar(data_population_hospitalbeds,x='State / Union Territory',y='TotalCases',height=600,color = 'TotalCases')
fig.show()
fig = px.bar(data_population_hospitalbeds,x='State / Union Territory',y='TotalHospitals_inState', height=600, color = 'TotalCases')
fig.show()

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
data = data.drop(labels='Sno',axis=1)
data = data.drop(labels='Time',axis=1)
data.Date = pd.to_datetime(data.Date)
data.Date = data.Date.astype(int)
data.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['State/UnionTerritory'] = le.fit_transform(data['State/UnionTerritory'])
Y = data['Deaths']
X = data.drop(['Deaths'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
print(y_test)

d = {'Date': ['28/03/20'], 'State/UnionTerritory': 'Karnataka','ConfirmedIndianNational':55,'ConfirmedForeignNational':0,'Cured':3,'TotalCases':55,'ActiveCases':52 }
df = pd.DataFrame(data=d)
df