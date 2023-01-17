# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

testsC = pd.read_csv('/kaggle/input/covid19-tests-conducted-by-country/Tests_conducted_23May2020.csv')
print(testsC.shape)

#POP Milyon değerinde bölge popülasyonu
#DP Hastalık Yaygınlığı
#test_Ratio Pozitif Testler
testsC['pop'] = testsC['Tested']/testsC['Tested\u2009/millionpeople']
testsC['dp'] = testsC['Positive']/testsC['pop']
testsC['test_Ratio'] =testsC['Tested']/testsC['Positive']
#Plotly grafiği importladık
#testsC tablosunu sıraladık
import plotly.graph_objs as go
testsC.sort_values(by=['Tested\u2009/millionpeople'], ascending=False, inplace=True)
trace1 = go.Bar(
                x = testsC['Country'],
                y = testsC['Tested\u2009/millionpeople'],
                name = "Disease Prevalence/Testing Ratio",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = testsC['Country'])

data = [trace1]
             
layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)
iplot(fig)

#Ülkelerin milyon popülasyon içerisinde dağılım grafiği yayılımı
#Milyon popülasyon içerisinde pozitif sonuçlanan testlerin yaygınlığı
trace1 = go.Scatter(
                    y = testsC['Tested\u2009/millionpeople'],
                    x = testsC['dp'],
                    mode = "markers",
                    name = "Country",
                    marker = dict(color = 'rgba(255, 50, 50, 0.8)'),
                    text= testsC['Country'])

data = [trace1]
layout = dict(title = 'Milyon Popülasyona karşılık Hastalık Yaygınlığı',
              xaxis= dict(title= 'Hastalık Yaygınlığı (Pozitif Testler/Milyon Nüfus' ,ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Milyon popülasyona göre yapılan test',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)
iplot(fig)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Train Csv dosyası aktarılıyor.
train = pd.read_csv("/kaggle/input/traincsv/train.csv",parse_dates=['Date'])
                    
train.tail()
complete_data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])
#Csv içindeki dosyaların sütunları tekrar adlandırılıyor.
complete_data = complete_data.rename(columns = {'Province/State': 'Bolge_Sehir', 'Country/Region': 'Ulke'})
#Tablo içerisine kullanılacak gerekli hesaplamalar yapılıyor.
complete_data['Active'] = complete_data['Confirmed'] - complete_data['Deaths'] - complete_data['Recovered']
#complede_data'dan aldığımız değerleri Date ve Confirmed'lere göre sıralıyoruz.
complete_data.sort_values(by=['Date','Confirmed'], ascending=False).head()
complete_data.info()
complete_data.sort_values(by=['Date'], ascending=False).tail()
#map_covid verilerini grupluyoruz.
map_covid = train.groupby(['Date', 'Country_Region'])['ConfirmedCases'].sum().reset_index()
map_covid['Date'] = map_covid['Date'].dt.strftime('%m/%d/%Y')
map_covid['size'] = map_covid['ConfirmedCases'].pow(0.3) * 3.5

fig = px.scatter_geo(map_covid, locations="Country_Region", locationmode='country names', 
                     color="ConfirmedCases", size='size', hover_name="Country_Region", 
                     range_color=[1,100],
                     projection="natural earth", animation_frame="Date", 
                     title='COVID-19: Dünya Üzerinde Onaylanmış Vakalar', color_continuous_scale="tealrose")
fig.show()
#df değişkenine covid 19 india csv sini tanımlıyoruz.
df=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
df
#Bazı sütunları tablodan kaldırıyoruz.
df=df.drop("ConfirmedIndianNational",axis=1)
df=df.drop("ConfirmedForeignNational",axis=1)
df
df.keys()
#Belirli bir bölge üzerinde işlem yapıyoruz.
new_data = df[df["State/UnionTerritory"] == "Kerala"]
new_data
#new_data[Date] değişkenine tarihi formatıyla birlikte tanımlıyoruz.
new_data['Date'] = pd.to_datetime(new_data.Date, format='%d/%m/%y').astype(str)
#0 ile 15 arasındaki index'de tarihler veriliyor.
new_data[0:15]
#Tarih formatında yer alan - silinerek  format değiştiriliyor.
new_data['Date']=new_data['Date'].str.replace("-","")
#Değiştirilen formata ait veriler veriliyor.
new_data['Date'][0:10]
#Tarih ve Vaka sayıları tanımlanıyor.
x=new_data["Date"]
y=new_data["Confirmed"]
#new_data tablosundaki değerleri lineer regresyonda uygulamak için x'e tanımlıyoruz.
x=new_data.iloc[:,0:1].values
lr=LinearRegression()
lr.fit(x,y)
#x değerini lineer regresyon ile birlikte y_predict'e tanımlıyoruz 
y_predict = lr.predict(x)
y_predict
#TABLOMUZU YAZDIRIYORUZ
plt.figure(figsize=(25,15))
plt.scatter(x,y)
plt.plot(x, y_predict, color="orange")
plt.xlabel("Tarih")
plt.ylabel("Vaka Sayısı")
plt.show()