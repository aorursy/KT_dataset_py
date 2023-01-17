import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import datetime as dt

from datetime import timedelta

from sklearn.preprocessing import PolynomialFeatures 

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing

from sklearn.metrics import mean_squared_error,r2_score

import statsmodels.api as sm

from fbprophet import Prophet

!pip install plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots
covid=pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

covid.head()
covid_turkey=covid[covid['Country/Region']=="Turkey"]

print(covid_turkey)

#print(covid_turkey.head())

covid_turkey["ObservationDate"]=pd.to_datetime(covid_turkey["ObservationDate"])
#Veriyi Tarihe göre sıralıyoruz

turkey_datewise=covid_turkey.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

turkey_datewise["WeekofYear"]=turkey_datewise.index.weekofyear
turkey_datewise["Days Since"]=(turkey_datewise.index-turkey_datewise.index[0])

turkey_datewise["Days Since"]=turkey_datewise["Days Since"].dt.days
No_Lockdown=covid_turkey[covid_turkey["ObservationDate"]<pd.to_datetime("2020-03-21")]

Lockdown_1=covid_turkey[(covid_turkey["ObservationDate"]>=pd.to_datetime("2020-03-21"))&(covid_turkey["ObservationDate"]<pd.to_datetime("2020-04-15"))]

Lockdown_2=covid_turkey[(covid_turkey["ObservationDate"]>=pd.to_datetime("2020-04-15"))&(covid_turkey["ObservationDate"]<pd.to_datetime("2020-05-04"))]

Lockdown_3=covid_turkey[(covid_turkey["ObservationDate"]>=pd.to_datetime("2020-05-04"))&(covid_turkey["ObservationDate"]<pd.to_datetime("2020-05-19"))]



Lockdown_4=covid_turkey[(covid_turkey["ObservationDate"]>=pd.to_datetime("2020-05-19"))&(covid_turkey["ObservationDate"]<pd.to_datetime("2020-06-30"))]



Lockdown_5=covid_turkey[(covid_turkey["ObservationDate"]>=pd.to_datetime("2020-06-30"))]



No_Lockdown_datewise=No_Lockdown.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

Lockdown_1_datewise=Lockdown_1.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

Lockdown_2_datewise=Lockdown_2.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

Lockdown_3_datewise=Lockdown_3.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

Lockdown_4_datewise=Lockdown_4.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

Lockdown_5_datewise=Lockdown_5.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
covid["ObservationDate"]=pd.to_datetime(covid["ObservationDate"])

grouped_country=covid.groupby(["Country/Region","ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
grouped_country["Active Cases"]=grouped_country["Confirmed"]-grouped_country["Recovered"]-grouped_country["Deaths"]

grouped_country["log_confirmed"]=np.log(grouped_country["Confirmed"])

grouped_country["log_active"]=np.log(grouped_country["Active Cases"])
print("Toplam Vaka Sayısı",turkey_datewise["Confirmed"].iloc[-1])

print("Toplam İyileşen Sayısı",turkey_datewise["Recovered"].iloc[-1])

print("Toplam Ölü Sayısı",turkey_datewise["Deaths"].iloc[-1])

print("Aktiv Vaka Sayısı",turkey_datewise["Confirmed"].iloc[-1]-turkey_datewise["Recovered"].iloc[-1]-turkey_datewise["Deaths"].iloc[-1])

print("Toplamda Kapatılan Vaka Sayısı",turkey_datewise["Recovered"].iloc[-1]+turkey_datewise["Deaths"].iloc[-1])

print("Ortalama Günllük Vaka Sayısı",round(turkey_datewise["Confirmed"].iloc[-1]/turkey_datewise.shape[0]))

print("Ortalama Günllük İyileşen Vaka Sayısı",round(turkey_datewise["Recovered"].iloc[-1]/turkey_datewise.shape[0]))

print("Ortalama Günllük Ölü Sayısı",round(turkey_datewise["Deaths"].iloc[-1]/turkey_datewise.shape[0]))

print("Son 24 Saatte Kayitedilen Vaka Sayısı",turkey_datewise["Confirmed"].iloc[-1]-turkey_datewise["Confirmed"].iloc[-2])

print("Son 24 Saatte Kayitedilen İyileşmiş Vaka Sayısı",turkey_datewise["Recovered"].iloc[-1]-turkey_datewise["Recovered"].iloc[-2])

print("Son 24 Saatte Kayitedilen Ölü Vaka Sayısı",turkey_datewise["Deaths"].iloc[-1]-turkey_datewise["Deaths"].iloc[-2])
fig=px.bar(x=turkey_datewise.index,y=turkey_datewise["Confirmed"]-turkey_datewise["Recovered"]-turkey_datewise["Deaths"],

           color_discrete_sequence=["red"])

fig.update_layout(title="Aktiv Vaka Sayısı dağılımı",

                  xaxis_title="Tarih",yaxis_title="Vaka Sayısı")

fig.show()
fig=px.bar(x=turkey_datewise.index,y=turkey_datewise["Recovered"]+turkey_datewise["Deaths"],

          color_discrete_sequence=["red"])

fig.update_layout(title="Kapatılan Vaka Sayısı",

                  xaxis_title="Tarih",yaxis_title="Vaka Sayısı")

fig.show()
fig=px.bar(x=turkey_datewise.index,y=turkey_datewise["Recovered"],

          color_discrete_sequence=["red"])

fig.update_layout(title="İyişeşen Vaka Sayısı",

                  xaxis_title="Tarih",yaxis_title="İyileşen Vaka Sayısı")

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=turkey_datewise.index, y=turkey_datewise["Confirmed"],

                    mode='lines+markers',

                    name='Vaka Sayısı'))

fig.add_trace(go.Scatter(x=turkey_datewise.index, y=turkey_datewise["Recovered"],

                    mode='lines+markers',

                    name='İyileşen Sayısı'))

fig.add_trace(go.Scatter(x=turkey_datewise.index, y=turkey_datewise["Deaths"],

                    mode='lines+markers',

                    name='Ölü Sayısı'))

fig.update_layout(xaxis_title="Tarih",yaxis_title="Sayı",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
df= pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df.head()
from sklearn.preprocessing import LabelEncoder

from plotly.offline import iplot, init_notebook_mode

import math

import bokeh 

import matplotlib.pyplot as plt

import plotly.express as px

from urllib.request import urlopen

import json

from dateutil import parser

from bokeh.layouts import gridplot

from bokeh.plotting import figure, show, output_file

from bokeh.layouts import row, column

from bokeh.resources import INLINE

from bokeh.io import output_notebook

from bokeh.models import Span

import warnings

warnings.filterwarnings("ignore")

output_notebook(resources=INLINE)

le=LabelEncoder()







df.rename(columns={'Country/Region': 'Country', 'ObservationDate': 'Date'}, inplace=True)

df = df.fillna('unknown')

df['Country'] = df['Country'].str.replace('US','United States')

df['Country'] = df['Country'].str.replace('UK','United Kingdom') 

df['Country'] = df['Country'].str.replace('Mainland China','China')

df['Code']=le.fit_transform(df['Country'])

virus_data = df

print(virus_data.head())

print(len(virus_data))



top_country = virus_data.loc[virus_data['Date'] == virus_data['Date'].iloc[-1]]

top_country = top_country.groupby(['Code','Country'])['Confirmed'].sum().reset_index()

top_country = top_country.sort_values('Confirmed', ascending=False)

top_country = top_country[:50]

top_country_codes = top_country['Country']

top_country_codes = list(top_country_codes)

#print(top_country)



countries = virus_data[virus_data['Country'].isin(top_country_codes)]

countries_day = countries.groupby(['Date','Code','Country'])['Confirmed','Deaths','Recovered'].sum().reset_index()

#print(countries_day)





exponential_line_x = []

exponential_line_y = []

for i in range(16):

    exponential_line_x.append(i)

    exponential_line_y.append(i)

    

    

###############

russia = countries_day.loc[countries_day['Code']==168]



new_confirmed_cases_russia = []

new_confirmed_cases_russia.append( list(russia['Confirmed'])[0] - list(russia['Deaths'])[0] 

                           - list(russia['Recovered'])[0] )



for i in range(1,len(russia)):



    new_confirmed_cases_russia.append( list(russia['Confirmed'])[i] - 

                                     list(russia['Deaths'])[i] - 

                                     list(russia['Recovered'])[i])

###############





###############

china = countries_day.loc[countries_day['Code']==43]



new_confirmed_cases_china = []

new_confirmed_cases_china.append( list(china['Confirmed'])[0] - list(china['Deaths'])[0] 

                           - list(china['Recovered'])[0] )



for i in range(1,len(china)):



    new_confirmed_cases_china.append( list(china['Confirmed'])[i] - 

                                     list(china['Deaths'])[i] - 

                                     list(china['Recovered'])[i])

###############









##############

italy = countries_day.loc[countries_day['Code']==102]



new_confirmed_cases_ita = []

new_confirmed_cases_ita.append( list(italy['Confirmed'])[0] - list(italy['Deaths'])[0] 

                           - list(italy['Recovered'])[0] )



for i in range(1,len(italy)):

    

    new_confirmed_cases_ita.append( list(italy['Confirmed'])[i] - 

                                  list(italy['Deaths'])[i] - 

                                  list(italy['Recovered'])[i])

####################











#################

turkey = countries_day.loc[countries_day['Code']== 206]



new_confirmed_cases_turkey = []

new_confirmed_cases_turkey.append( list(turkey['Confirmed'])[0] - list(turkey['Deaths'])[0] 

                           - list(turkey['Recovered'])[0] )



for i in range(1,len(turkey)):

    

    new_confirmed_cases_turkey.append( list(turkey['Confirmed'])[i] - 

                                     list(turkey['Deaths'])[i] - 

                                    list(turkey['Recovered'])[i])

##################









##################

spain = countries_day.loc[countries_day['Code']==188]



new_confirmed_cases_spain = []

new_confirmed_cases_spain.append( list(spain['Confirmed'])[0] - list(spain['Deaths'])[0] 

                           - list(spain['Recovered'])[0] )



for i in range(1,len(spain)):

    

    new_confirmed_cases_spain.append( list(spain['Confirmed'])[i] - 

                                     list(spain['Deaths'])[i] - 

                                    list(spain['Recovered'])[i])

###################









##################

us = countries_day.loc[countries_day['Code']==211]



new_confirmed_cases_us = []

new_confirmed_cases_us.append( list(us['Confirmed'])[0] - list(us['Deaths'])[0] 

                           - list(us['Recovered'])[0] )



for i in range(1,len(us)):

    

    new_confirmed_cases_us.append( list(us['Confirmed'])[i] - 

                                     list(us['Deaths'])[i] - 

                                    list(us['Recovered'])[i])

###################









####################

german = countries_day.loc[countries_day['Code']==77]



new_confirmed_cases_german = []

new_confirmed_cases_german.append( list(german['Confirmed'])[0] - list(german['Deaths'])[0] 

                           - list(german['Recovered'])[0] )



for i in range(1,len(german)):

    

    new_confirmed_cases_german.append( list(german['Confirmed'])[i] - 

                                     list(german['Deaths'])[i] - 

                                    list(german['Recovered'])[i])

########################







#***

p1=figure(plot_width=800, plot_height=550, title="Ülkelere Göre Aktiv ve Pasiv Vakalar")

p1.grid.grid_line_alpha=0.3

p1.xaxis.axis_label = 'Vaka Sayısının Toplamı'

p1.yaxis.axis_label = 'Aktiv Vaka Sayısı Toplamı'

#****





p1.line(exponential_line_x, exponential_line_y, line_dash="4 4", line_width=1)



p1.line(np.log(list(russia['Confirmed'])), np.log(new_confirmed_cases_russia), color='darkturquoise', 

        legend_label='Russia', line_width=2)

p1.circle(np.log(list(russia['Confirmed'])[-1]), np.log(new_confirmed_cases_russia[-1]), size=5)



p1.line(np.log(list(china['Confirmed'])), np.log(new_confirmed_cases_china), color='orange', 

        legend_label='China', line_width=2)

p1.circle(np.log(list(china['Confirmed'])[-1]), np.log(new_confirmed_cases_china[-1]), size=5)



p1.line(np.log(list(italy['Confirmed'])), np.log(new_confirmed_cases_ita), color='yellow', 

        legend_label='Italiya', line_width=2)

p1.circle(np.log(list(italy['Confirmed'])[-1]), np.log(new_confirmed_cases_ita[-1]), size=5)



p1.line(np.log(list(turkey['Confirmed'])), np.log(new_confirmed_cases_turkey), color='red', 

        legend_label='Turkey', line_width=2)

p1.circle(np.log(list(turkey['Confirmed'])[-1]), np.log(new_confirmed_cases_turkey[-1]), size=5)



p1.line(np.log(list(spain['Confirmed'])), np.log(new_confirmed_cases_spain), color='brown', 

        legend_label='İspanya', line_width=2)

p1.circle(np.log(list(spain['Confirmed'])[-1]), np.log(new_confirmed_cases_spain[-1]), size=5)



p1.line(np.log(list(us['Confirmed'])), np.log(new_confirmed_cases_us), color='green', 

        legend_label='ABD', line_width=2)

p1.circle(np.log(list(us['Confirmed'])[-1]), np.log(new_confirmed_cases_us[-1]), size=5)



p1.line(np.log(list(german['Confirmed'])), np.log(new_confirmed_cases_german), color='black', 

        legend_label='Almanya', line_width=2)

p1.circle(np.log(list(german['Confirmed'])[-1]), np.log(new_confirmed_cases_german[-1]), size=5)



p1.legend.location = "bottom_right"



show(p1)

from sklearn.linear_model import Perceptron

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.svm import NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier



from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import KNeighborsRegressor



from sklearn.svm import SVC

import plotly.offline as py

import plotly.express as px

from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

"""

X = virus_data.iloc[:,5:7].values

y = virus_data.iloc[:, 6].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

accuracies.mean()

"""


X = virus_data.iloc[:, [ 5, 7]].values

y = virus_data.iloc[:, 6].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)





classifier=BaggingClassifier(random_state=0)

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

acc=accuracy_score(y_test, y_pred)

print(acc)


X = virus_data.iloc[:, [ 5, 7]].values

y = virus_data.iloc[:, 6].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)





classifier=GaussianNB()

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

acc=accuracy_score(y_test, y_pred)

print(acc)
X = virus_data.iloc[:, [ 5, 7]].values

y = virus_data.iloc[:, 6].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)





classifier=BernoulliNB()

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

acc=accuracy_score(y_test, y_pred)

print(acc)
X = virus_data.iloc[:, [ 6, 7]].values

y = virus_data.iloc[:, 6 ].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)





classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

acc=accuracy_score(y_test, y_pred)

print(acc)
X = virus_data.iloc[:, [ 5, 7]].values

y = virus_data.iloc[:, 6].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)





classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

acc=accuracy_score(y_test, y_pred)

print(acc)