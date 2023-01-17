import numpy as np

import pandas as pd

import os



urunler = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

kategoriler = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

satislar = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

magazalar = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')







satislar.head()

#kategoriler.head()

magazalar.head()
import plotly.offline as py

import plotly.express as px

from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot



thveri = satislar   # Tahmin verileri

thveri = thveri.fillna(0)

thveri_grp = thveri.groupby("date")[["item_id","item_cnt_day"]].sum().reset_index()

thveri_cntday = thveri_grp.loc[:,["date","item_cnt_day"]]

th_data = thveri_cntday

th_data.columns = ['ds','y']    #ds: dates  -  y: values

th_data.head()
urunModel=Prophet()

urunModel.fit(th_data)

ozellik1=urunModel.make_future_dataframe(periods=6,freq='M')   #  Bir sonraki ay olan Kasım 2015'in de içinde bulunduğu gelecek 6 ayın tahmin verileri için özellik dataframe'i

tahmin1=urunModel.predict(ozellik1)

tahmin1



#fig = plot_plotly(m, tahmin)

#py.iplot(fig) 



fig = urunModel.plot(tahmin1,xlabel='Aylar',ylabel='Aylara Göre Satılan Toplam Ürün Miktarı', figsize=(20,5))   # Son 6 ay tahminleri içermektedir
ft=thveri

def mask(df, key, value):

    return df[df[key] == value]

pd.DataFrame.mask = mask



id=21                        #        Mağaza id'si "21" seçilsin



magaza=magazalar.loc[magazalar['shop_id'] == id, 'shop_name'].iloc[0]



ft2=ft.mask('shop_id', id)   #        Bu id'ye göre filtreledim ---- Önemli: Bazı mağazaların 2015 Ekim'e kadar satış verileri yok

ft2.tail(10)
print("Secilen magaza adı: ",magaza)




magazaTahmin_grp = ft2.groupby("date")["item_cnt_day"].sum().reset_index()

magazaTahmin_cntday = magazaTahmin_grp.loc[:,["date","item_cnt_day"]]

magazaTahmin_data = magazaTahmin_cntday

magazaTahmin_data.columns = ['ds','y']    #ds: dates  -  y: values olmakta

magazaTahmin_data.head()



modelMagaza=Prophet()

modelMagaza.fit(magazaTahmin_data)

dFozellik2=modelMagaza.make_future_dataframe(periods=6,freq='M')   #  Bir sonraki ay olan Kasım 2015'in de içinde bulunduğu gelecek 6 ayın tahmin verileri için özellik dataframe'i

tahmin2=modelMagaza.predict(dFozellik2)

tahmin2



#fig2 = plot_plotly(modelMagaza, tahmin2)

#py.iplot(fig2) 







fig2=modelMagaza.plot(tahmin2,xlabel='Aylar',ylabel= magaza + "   Mağazasının Aylık Satış Miktarı", figsize=(20,5))   # Magaza yukarıdan değiştirilebilir
from fbprophet.diagnostics import cross_validation

df_cv = cross_validation(modelMagaza, initial='730 days', period='30 days', horizon = '180 days')    # Yukarıdaki tahmin periyodunu 6 belirlediğimden horizon = 6*30 olmalı

df_cv.head()
from fbprophet.diagnostics import performance_metrics

df_p = performance_metrics(df_cv)

df_p.head()

from fbprophet.plot import plot_cross_validation_metric

fig = plot_cross_validation_metric(df_cv, metric='mape')