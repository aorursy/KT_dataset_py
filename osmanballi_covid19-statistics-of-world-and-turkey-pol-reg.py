import random

from datetime import timedelta



# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import folium







# converter

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()   



# hide warnings

import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

df['Date']=pd.to_datetime(df['Date'])

df.head(10)

Turkey_df=df[df['Country/Region']=='Turkey']



temp = df[df['Date'] == max(df['Date'])]



m = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=4, zoom_start=1)



for i in range(0, len(temp)):

    folium.Circle(

        location=[temp.iloc[i]['Lat'], temp.iloc[i]['Long']],

        color='crimson', fill='crimson',

        tooltip =   '<li><bold>Country : '+str(temp.iloc[i]['Country/Region'])+

                    '<li><bold>Province : '+str(temp.iloc[i]['Province/State'])+

                    '<li><bold>Confirmed : '+str(temp.iloc[i]['Confirmed'])+

                    '<li><bold>Deaths : '+str(temp.iloc[i]['Deaths']),

        radius=int(temp.iloc[i]['Confirmed'])**1.1).add_to(m)

m
fig = px.choropleth(df, locations="Country/Region", locationmode='country names', color=np.log(df["Confirmed"]), 

                    hover_name="Country/Region", animation_frame=df["Date"].dt.strftime('%Y-%m-%d'),

                    title='Cases over time', color_continuous_scale=px.colors.sequential.Magenta)

fig.update(layout_coloraxis_showscale=False)

fig.show()
# Country wise

# ============



# getting latest values

country_wise = df[df['Date']==max(df['Date'])].reset_index(drop=True).drop('Date', axis=1)



# group by country

country_wise = country_wise.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()



# per 100 cases

country_wise['Deaths / 100 Cases'] = round((country_wise['Deaths']/country_wise['Confirmed'])*100, 2)

country_wise['Recovered / 100 Cases'] = round((country_wise['Recovered']/country_wise['Confirmed'])*100, 2)

country_wise['Deaths / 100 Recovered'] = round((country_wise['Deaths']/country_wise['Recovered'])*100, 2)



cols = ['Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered']

country_wise[cols] = country_wise[cols].fillna(0)



display(country_wise.head())

display(country_wise[country_wise['Country/Region']=='Turkey'])
# confirmed - deaths

fig_c = px.bar(country_wise.sort_values('Confirmed').tail(15), x="Confirmed", y="Country/Region", 

               text='Confirmed', orientation='h', color_discrete_sequence = ['#a3de83'])

fig_d = px.bar(country_wise.sort_values('Deaths').tail(15), x="Deaths", y="Country/Region", 

               text='Deaths', orientation='h', color_discrete_sequence = ['#f38181'])











# plot

fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,

                    subplot_titles=('Confirmed cases', 'Deaths reported'))



fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)



fig.update_layout(height=500)
from IPython.core.display import HTML

HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1571387"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
from datetime import date

Turkey_df=Turkey_df[Turkey_df['Date']>date(2020,3,11)]

plt.figure(figsize=(15,5))

sns.barplot(x=Turkey_df['Date'].dt.strftime('%Y-%m-%d'), y=Turkey_df["Confirmed"]-Turkey_df["Recovered"]-Turkey_df["Deaths"])

plt.title("Distribution Plot for Active Cases Cases over Date")

plt.xticks(rotation=90)
# color pallette

cnf, dth, rec = '#ff7f00', '#ff2e63', '#21bf73'



temp = Turkey_df.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)



tm = temp.melt(id_vars="Date", value_vars=['Confirmed', 'Deaths', 'Recovered'])

fig = px.treemap(tm, path=["variable"], values="value", height=225, width=1200,

                 color_discrete_sequence=[cnf, rec, dth])

fig.data[0].textinfo = 'label+text+value'

fig.show()


plt.figure(figsize=(16, 9))

plt.plot(Turkey_df['Date'], Turkey_df['Confirmed'])

plt.plot(Turkey_df['Date'], Turkey_df['Deaths'])

plt.plot(Turkey_df['Date'], Turkey_df['Recovered'])

plt.title('Coronavirus Cases in Turkey', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('Cases', size=30)

plt.legend(['Confirmed', 'Deaths', 'Recovered'], prop={'size': 20})

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
confirmed=Turkey_df["Confirmed"]
import numpy as np

confirmed=Turkey_df["Confirmed"].values

gun_sayisi=len(Turkey_df[Turkey_df['Date']>date(2020,3,11)])

x=np.arange(0 , gun_sayisi)

gunluk_vaka=[]

for n in x:

    gunluk_vaka.append(confirmed[x-n]-confirmed[x-n-1])

    if n == 0:

        gunluk_vaka[0][0]=1

        break

print(gunluk_vaka)

x=x.tolist()

gunluk_vaka=gunluk_vaka[0].tolist()

x=pd.Series(x)

gunluk_vaka=pd.Series(gunluk_vaka)

print(x.shape)

print(gunluk_vaka.shape)
import numpy as np

gun_sayisi=len(Turkey_df[Turkey_df['Date']>date(2020,3,11)])



x=x.values.reshape(-1,1)

y=gunluk_vaka.values.reshape(-1,1)

print(x.shape)

print(y.shape)

plt.figure(figsize=(16, 9))

plt.scatter(x,y)

plt.xlabel("Gün Sayısı")

plt.ylabel("Vaka Sayısı")

plt.title("Günlük Vaka Sayısı")

plt.show()
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

Poly_reg=PolynomialFeatures(degree=5) #7. dereceden polinom 

x_poly=Poly_reg.fit_transform(x)

Lin_reg = LinearRegression()

Lin_reg.fit(x_poly,y)

print(x_poly[:5])
plt.figure(figsize=(16, 9))

plt.scatter(x,y)

plt.xlabel("Gün Sayısı")

plt.ylabel("Vaka Sayısı")

plt.title("Regresyon Modeli")

y_pred=Lin_reg.predict(x_poly)

plt.plot(x,y_pred,color="green",label="Polinom Linner Regresyon Model")

plt.legend()

plt.show()

print(Lin_reg.intercept_)

print(Lin_reg.coef_)