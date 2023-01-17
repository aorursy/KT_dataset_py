# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
covid=pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
covid.head(-5)
eyaletler=covid['State/UnionTerritory'].unique()
eyaletler
covid.info()
covid.isnull().sum()
covid.head(20)
ss = covid.loc[(covid['State/UnionTerritory'] == 'Punjab')]
ss.head(20)
import seaborn as sns
sns.countplot(x='ConfirmedIndianNational',data=ss)
import plotly.offline as pyoff
import plotly.graph_objs as pygo
Iyilesen_sayisi = pygo.Scatter(x=ss['Date'], y=ss['Cured'], name= 'Iyilesenler')
Olen_sayisi = pygo.Scatter(x=ss['Date'], y=ss['Deaths'], name= 'Vefat Edenler')
pyoff.iplot([Iyilesen_sayisi,Olen_sayisi])
sem=ss[['Confirmed']]
sem = sem.values
Dizi_boyutu = int(len(sem) * 0.80)
Test_sayisi = len(sem) - Dizi_boyutu
dizi, test = sem[0:Dizi_boyutu,:], sem[Dizi_boyutu:len(sem),:]
print(len(dizi), len(test))
def datasetim(dataset, ileri=1):
    dataX, dataY = [], [] 
    for i in range(len(dataset)-ileri-1):
        a = dataset[i:(i+ileri), 0]
        dataX.append(a)
        dataY.append(dataset[i + ileri, 0])
    return np.array(dataX), np.array(dataY)
ileri = 2
diziX, diziY = datasetim(dizi, ileri=ileri)
testX, testY = datasetim(test, ileri=ileri)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(diziX,diziY)
tahmin=model.predict(testX)
ss = pd.DataFrame({'Gercek': testY.flatten(), 'Tahmini': tahmin.flatten()})
ss
ss.plot(kind='bar',figsize=(16,16))
#Yaş aralıkları
yas_gruplari = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')
yas_gruplari.head(10)
yas_gruplari.columns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set()
sns.set(style="whitegrid")
ax = sns.barplot(x="AgeGroup", y="TotalCases", data=yas_gruplari)  
covid['totalIndian'] = (covid['Cured']+covid['Deaths']+covid['Confirmed'])
covid.groupby('State/UnionTerritory')['totalIndian'].sum()
covid['State/UnionTerritory'].value_counts()
!pip install fbprophet
from fbprophet import Prophet
def gelecekdurumtahmini(ds, durum, gun):
    ds_durum = ds[ds['State/UnionTerritory']==durum]
    ds_model = ds_durum[['Date','Confirmed']].rename(columns={'Date': 'ds', 'Confirmed': 'y'})
    m = Prophet()
    m.fit(ds_model)
    gelecek =  m.make_future_dataframe(periods=gun)
    tahmin=m.predict(gelecek)
    figur1 = m.plot(tahmin)
    figur2 = m.plot_components(tahmin)
gelecekdurumtahmini(covid,'Punjab',20)