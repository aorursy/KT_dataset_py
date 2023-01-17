import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

%matplotlib inline

from subprocess import check_output

import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt # visualization library

from plotly.offline import init_notebook_mode, plot, iplot

import plotly as py

init_notebook_mode(connected=True) 

import plotly.graph_objs as go # plotly graphical object

import warnings            

warnings.filterwarnings("ignore")

import seaborn as sns

import statsmodels.api as sm

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#data upload

balata = pd.ExcelFile("/kaggle/input/brakediscdata/Balata.xlsx").parse('Sheet1')

print(balata.dtypes)

print(balata.describe())
balata.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in balata.columns]

balata["Replaced_Quantity"] = 1

print(balata.dtypes)
#Column Frequency Count

print(balata["Model1"].value_counts(dropna =False))  #model kodu bazında kırılım

print(balata["Dealer_Name"].value_counts(dropna =False))  #bayi bazında kırılım

print(balata["OFP"].value_counts(dropna =False))  #bayi bazında kırılım #parçano bazında kırılım
df1 = balata[["Model1","Mileage"]]

df1["index"] = np.arange(1,len(df1)+1)

df1 = df1.set_index('index')

df1 = pd.pivot_table(df1, values='Mileage',index='index',columns='Model1')



trace0 = go.Box(

    y=df1.NDE180L,

    name = 'NDE180L',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Box(

    y=df1.NRE180L,

    name = 'NRE180L',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

trace2 = go.Box(

    y=df1.ZRE181L,

    name = 'ZRE181L',

    marker = dict(

        color = 'rgb(10, 10, 10)',

    )

)

trace3 = go.Box(

    y=df1.ZRE210L,

    name = 'ZRE210L',

    marker = dict(

        color = 'rgb(10, 10, 10)',

    )

)

trace4 = go.Box(

    y=df1.ZWE211L,

    name = 'ZWE211L',

    marker = dict(

        color = 'rgb(10, 10, 10)',

    )

)

trace5 = go.Box(

    y=balata.Mileage,

    name = 'Total Mileage',

    marker = dict(

        color = 'rgb(70, 70, 70)',

    )

)

data = [trace0, trace1, trace2, trace3,trace4,trace5]

iplot(data)
df2 = balata[["Model","Mileage"]]



df2["index"] = np.arange(1,len(df2)+1)

df2 = df2.set_index('index')

df2 = pd.pivot_table(df2, values='Mileage',index='index',columns='Model')

df2=df2.rename(columns = {'AURIS TOURING SPORTS':'AURIS_TOURING'})

df2=df2.rename(columns = {'COROLLA S/D':'COROLLA_SEDAN'})



trace1 = go.Box(

    y=df2.AURIS,

    name = 'AURIS',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace2 = go.Box(

    y=df2.AURIS_TOURING,

    name = 'AURIS_TOURING',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

trace3 = go.Box(

    y=df2.COROLLA_SEDAN,

    name = 'COROLLA_SEDAN',

    marker = dict(

        color = 'rgb(10, 10, 10)',

    )

)

trace4 = go.Box(

    y=balata.Mileage,

    name = 'COROLLA S/D HYBRID',

    marker = dict(

        color = 'rgb(70, 70, 70)',

    )

)

trace5 = go.Box(

    y=balata.Mileage,

    name = 'Total Mileage',

    marker = dict(

        color = 'rgb(70, 70, 70)',

    )

)

data = [trace1, trace2, trace3, trace4,trace5]

iplot(data)
df = balata[["Model","Model1","OFP","Mileage"]]



sns.set(font_scale = 1.5)

sns.set_style("white")

sns.catplot(x='Model', y='Mileage',

                  hue="Model1",

                data=df, kind="box",

            height=6, aspect=3.3);



sns.set(font_scale = 1.5)

sns.set_style("white")

sns.catplot(x='Model', y='Mileage',

                  hue="OFP",

                data=df, kind="box",

            height=6, aspect=5.3);
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(balata[["Dealer_Code","Mileage","OFP","T1_Code","T2_Code"]].corr(), annot=True, linewidths=.15, fmt= '.1f',ax=ax)

plt.show()
# prepare data

df3 = balata[["Dealer_Code","Replaced_Quantity","Dealer_Name"]].groupby(["Dealer_Name"], as_index = False).sum().sort_values(by="Replaced_Quantity",ascending = False)

df3["index"] = np.arange(1,len(df3)+1)

df3 = df3.set_index('index')



x = df3.Dealer_Name

y = df3.Replaced_Quantity



fig = go.Figure()

fig.add_trace(go.Histogram(histfunc="sum", y=y, x=x, name="sum"))



fig.show()
#Workorder Date Yıl vs Ay Kırılımı

Workorder_TimeSeries_m = balata.set_index("WorkOrder_Date")

Workorder_TimeSeries_m = Workorder_TimeSeries_m.Replaced_Quantity.resample("M").sum()

Workorder_TimeSeries_m.plot()



Workorder_TimeSeries_y = balata.set_index("WorkOrder_Date")

Workorder_TimeSeries_y = Workorder_TimeSeries_y.Replaced_Quantity.resample("A").sum()

Workorder_TimeSeries_y.plot()





#Production Date Yıl vs Ay Kırılımı

Production_TimeSeries_m = balata.set_index("Production_Date")

Production_TimeSeries_m = Production_TimeSeries_m.Replaced_Quantity.resample("M").sum()

Production_TimeSeries_m.plot()



Production_TimeSeries_y = balata.set_index("Production_Date")

Production_TimeSeries_y = Production_TimeSeries_y.Replaced_Quantity.resample("A").sum()

Production_TimeSeries_y.plot()
#Delivery Date Yıl vs Ay Kırılımı

Delivery_TimeSeries_m = balata.set_index("Delivery_Date")

Delivery_TimeSeries_m = Delivery_TimeSeries_m.Replaced_Quantity.resample("M").sum()

Delivery_TimeSeries_m.plot()



Delivery_TimeSeries_y = balata.set_index("Delivery_Date")

Delivery_TimeSeries_y = Delivery_TimeSeries_y.Replaced_Quantity.resample("A").sum()

Delivery_TimeSeries_y.plot()
speech_analytics = pd.ExcelFile("/kaggle/input/voc-vs-dealer-inspection-text/VOCvsDealer Inspection.xlsx").parse('Sheet1')
df4 = speech_analytics[["VOC"]].apply(lambda x: x.astype(str).str.lower())



text1 = " ".join(review for review in df4.VOC)

print ("There are {} words in the combination of all review.".format(len(text1)))



# Create stopword list:

stopwords = set(STOPWORDS)

stopwords.update(["geliyor","geli","yor","seyi","hali","nde","islik","gi","derken"])



# Generate a word cloud image

wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text1)



# Display the generated image:

# the matplotlib way:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
df5 = speech_analytics[["Dealer_Inspection"]].apply(lambda x: x.astype(str).str.lower())



text2 = " ".join(review for review in df5.Dealer_Inspection)

print ("There are {} words in the combination of all review.".format(len(text2)))



# Create stopword list:

stopwords = set(STOPWORDS)

stopwords.update(["geliyor","geli","yor","seyi","hali","nde","islik","gi","derken"])



# Generate a word cloud image

wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text2)



# Display the generated image:

# the matplotlib way:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
y = balata.set_index("WorkOrder_Date")

y = y.Replaced_Quantity.resample("W").sum()



y.plot(figsize=(19, 4))

plt.show()
from pylab import rcParams

rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(y, model='additive')

fig = decomposition.plot()

plt.show()
!pip install pmdarima

import pmdarima as pm



# Seasonal - fit stepwise auto-ARIMA

smodel = pm.auto_arima(y, start_p=1, start_q=1,

                         test='adf',

                         max_p=3, max_q=3, m=12,

                         start_P=0, seasonal=True,

                         d=None, D=1, trace=True,

                         error_action='ignore',  

                         suppress_warnings=True, 

                         stepwise=True)



smodel.summary()
mod = sm.tsa.statespace.SARIMAX(y,

                                order=(1, 0, 2),

                                seasonal_order=(0, 1, 1, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
results.plot_diagnostics(figsize=(18, 8))

plt.show()
pred = results.get_prediction(start=pd.to_datetime('2019-02-03'), dynamic=False)

pred_ci = pred.conf_int()

ax = y['2013':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')

ax.set_ylabel('Retail_sold')

plt.legend()

plt.show()
y_forecasted = pred.predicted_mean

y_truth = y['2018-06-01':]

mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results.get_forecast(steps=48)

pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(14, 4))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Replaced_Quantity')

plt.legend()

plt.show()
pred_ci.head(48)
forecast = pred_uc.predicted_mean

forecast.head(48)