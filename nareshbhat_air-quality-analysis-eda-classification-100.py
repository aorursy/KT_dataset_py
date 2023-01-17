#loading

import pandas as pd

import numpy as np

import missingno as msno

# Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns

!pip install chart_studio

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')

#preprocessing 

from sklearn.preprocessing import StandardScaler, LabelEncoder

from collections import Counter

# Classification 

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
data = pd.read_csv("../input/air-quality-pre-and-post-covid19-pandemic/City_day.csv")

data['Date'] = pd.to_datetime(data['Date'])

data.head()
data.info()
msno.heatmap(data)
df1 = data.copy()

df1['PM2.5']=df1['PM2.5'].fillna((df1['PM2.5'].median()))

df1['PM10']=df1['PM10'].fillna((df1['PM10'].median()))

df1['NO']=df1['NO'].fillna((df1['NO'].median()))

df1['NO2']=df1['NO2'].fillna((df1['NO2'].median()))

df1['NOx']=df1['NOx'].fillna((df1['NOx'].median()))

df1['NH3']=df1['NH3'].fillna((df1['NH3'].median()))

df1['CO']=df1['CO'].fillna((df1['CO'].median()))

df1['SO2']=df1['SO2'].fillna((df1['SO2'].median()))

df1['O3']=df1['O3'].fillna((df1['O3'].median()))

df1['Benzene']=df1['Benzene'].fillna((df1['Benzene'].median()))

df1['Toluene']=df1['Toluene'].fillna((df1['Toluene'].median()))

df1['Xylene']=df1['Xylene'].fillna((df1['Xylene'].median()))

df1['AQI']=df1['AQI'].fillna((df1['AQI'].median()))

df1['Air_quality']=df1['Air_quality'].fillna('Moderate')
df = df1.copy()

df = df[df['Date'] <= ('01-01-2020')] 

df['Vehicular Pollution content'] = df['PM2.5']+df['PM10']+df['NO']+df['NO2']+df['NOx']+df['NH3']+df['CO']

df['Industrial Pollution content'] = df['SO2']+df['O3']+df['Benzene']+df['Toluene']+df['Xylene']

df = df.drop(['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',

       'O3','Benzene','Toluene','Xylene'],axis=1)

df.info()
def ploting(var):

    df[var].iplot(title=var,xTitle='Cities',yTitle=var, linecolor='black', )

    plt.show()

ploting('Vehicular Pollution content')

ploting('Industrial Pollution content')
def max_bar_plot(var):

    x1 = df[['City',var]].groupby(["City"]).median().sort_values(by = var,

    ascending = True).tail(10).iplot(kind='bar', xTitle='Cities',yTitle=var, 

                                     linecolor='black', title='{2} {1} {0}'.format(")",var,' Most polluted cities('))



p1 = max_bar_plot('Industrial Pollution content')

p2 = max_bar_plot('Vehicular Pollution content')
def min_bar_plot(var):

    x1 = df[['City',var]].groupby(["City"]).mean().sort_values(by = var,

    ascending = True).head(10).iplot(kind='bar', yTitle='Cities',xTitle=var, linecolor='black',title='{2} {1} {0}'.format(")",var,' Minimum polluted cities('))

p1 = min_bar_plot('Industrial Pollution content')

p2 = min_bar_plot('Vehicular Pollution content')
def al(var):

    cities = [var]

    filtered_city_day = df1[df1['Date'] <= '2020-04-01']

    AQI = filtered_city_day[filtered_city_day.City.isin(cities)][['City','Air_quality']]

    AQI[var] = AQI['Air_quality']

    k = AQI[var].value_counts()

    m = pd.DataFrame((round((k/sum(k))*100)))

    return m

c11 = al('Ahmedabad')

c22 = al('Delhi')

c33 = al('Kolkata')

c44 = al('Mumbai')

c55 = al('Bengaluru')

df_row = pd.concat([c11,c22,c33,c44,c55],axis=1)

df_row.iplot(kind='bar', align='center',xTitle='Satisfaction level', yTitle='percentage of satisfaction' ,linecolor='black', title='Satisfaction level of people(Pre COVID19)')
df = df1.copy()

df = df[df['Date'] > ('01-01-2020')] 

df['Vehicular Pollution content'] = df['PM2.5']+df['PM10']+df['NO']+df['NO2']+df['NOx']+df['NH3']+df['CO']

df['Industrial Pollution content'] = df['SO2']+df['O3']+df['Benzene']+df['Toluene']+df['Xylene']

df = df.drop(['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',

       'O3','Benzene','Toluene','Xylene'],axis=1)

df.info()
def ploting(var):

    df[var].iplot(title=var,xTitle='Cities',yTitle=var, linecolor='black', )

    plt.show()

ploting('Vehicular Pollution content')

ploting('Industrial Pollution content')
def max_bar_plot(var):

    x1 = df[['City',var]].groupby(["City"]).median().sort_values(by = var,

    ascending = True).tail(10).iplot(kind='bar', xTitle='Cities',yTitle=var, 

                                     linecolor='black', title='{2} {1} {0}'.format(")",var,' Most polluted cities('))



p1 = max_bar_plot('Industrial Pollution content')

p2 = max_bar_plot('Vehicular Pollution content')
def min_bar_plot(var):

    x1 = df[['City',var]].groupby(["City"]).mean().sort_values(by = var,

    ascending = True).head(10).iplot(kind='bar', yTitle='Cities',xTitle=var, linecolor='black',title='{2} {1} {0}'.format(")",var,' Minimum polluted cities('))

p1 = min_bar_plot('Industrial Pollution content')

p2 = min_bar_plot('Vehicular Pollution content')
def al(var):

    cities = [var]

    filtered_city_day = df1[df1['Date'] > '2020-04-01']

    AQI = filtered_city_day[filtered_city_day.City.isin(cities)][['City','Air_quality']]

    AQI[var] = AQI['Air_quality']

    k = AQI[var].value_counts()

    m = pd.DataFrame((round((k/sum(k))*100)))

    return m

c11 = al('Ahmedabad')

c22 = al('Delhi')

c33 = al('Kolkata')

c44 = al('Mumbai')

c55 = al('Bengaluru')

df_row = pd.concat([c11,c22,c33,c44,c55],axis=1)

df_row.iplot(kind='bar', align='center',xTitle='Satisfaction level', yTitle='percentage of satisfaction' ,linecolor='black', title='Satisfaction level of people(Post COVID19)')
categorical_attributes = list(df1.select_dtypes(include=['object']).columns)

print("categorical_attributes",categorical_attributes)

le=LabelEncoder()

df1['City']=le.fit_transform(df1['City'].astype(str))

df1['Air_quality']=le.fit_transform(df1['Air_quality'].astype(str))

df1.info()
cor = df1.corr()

cor.style.background_gradient(cmap='coolwarm')
y = df1["Air_quality"]

x = df1[['City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',

       'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
print('Classes and number of values in trainset',Counter(y_train))
from imblearn.over_sampling import SMOTE

oversample = SMOTE()

X_train,y_train = oversample.fit_resample(X_train,y_train)

print('Classes and number of values in trainset after SMOTE:',Counter(y_train))
cls=SVC()

cls.fit(X_train,y_train)

svmpred=cls.predict(X_test)

svmpred

cm=confusion_matrix(y_test,svmpred)

print("confussion matrix")

print(cm)

print("\n")

accuracy=accuracy_score(y_test,svmpred)

print("accuracy",accuracy*100)
rf = RandomForestClassifier(n_estimators=20, random_state=23)

rf.fit(X_train, y_train)

rf_predict=rf.predict(X_test)

rf_predict1=rf.predict(X_train)

rf_conf_matrix = confusion_matrix(y_test, rf_predict)

rf_acc_score = accuracy_score(y_test, rf_predict)

print("confussion matrix")

print(rf_conf_matrix)

print("\n")

print("accuracy",rf_acc_score*100)
gbc=XGBClassifier(learning_rate =0.01,n_estimators=100,max_depth=1,

                  min_child_weight=6,subsample=0.8,seed=13)

gbc.fit(X_train,y_train)

pred = gbc.predict(X_test)

xgb_conf_matrix = confusion_matrix(y_test, pred)

accuracy = accuracy_score(y_test, pred)*100

print("confussion matrix")

print(xgb_conf_matrix)

print("\n")

print("accuracy",accuracy)