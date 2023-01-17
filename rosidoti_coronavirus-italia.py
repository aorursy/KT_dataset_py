#import basic python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#read in a dataframe from remote url
git_hub_url="https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
df = pd.read_csv(git_hub_url,  parse_dates=True, index_col='data',sep=',')
df.plot(figsize=(18,8))
df.drop(columns=["ricoverati_con_sintomi","tamponi"]).plot(figsize=(18,8))
#remove not useful column and show last 3 days
df=df.drop(columns=["stato","note_it","note_en"])
df_delta=df.drop(df.columns,axis=1)

df.tail(6)
#let's define a couple of python functions useful for next steps
def delta(dfxx,df1):
  for c in dfxx.columns:
    new_name="delta_"+c
    df1[new_name]=(dfxx[c]-dfxx[c].shift()).fillna(0)
  return df1
def plot(dfx,cols=None,sizex=(18,8)):
    dff=dfx
    if cols:  
      dff=dfx[cols]
    dff.plot(figsize=sizex)
plot(df,["deceduti","dimessi_guariti","terapia_intensiva"])
plot(df,["totale_positivi","totale_casi"])
df_delta=delta(df,df_delta)
df_delta.tail(6)
plot(df_delta,["delta_deceduti","delta_dimessi_guariti","delta_terapia_intensiva","delta_totale_positivi","delta_totale_casi"])
#df_delta=df_delta.drop(columns=["delta_tamponi"],axis=1)
plot(df_delta,["delta_deceduti","delta_dimessi_guariti","delta_terapia_intensiva"])
#df_delta.plot(figsize=(18,8))
plot(df_delta,["delta_totale_positivi","delta_totale_casi"])
from fbprophet import Prophet
pdf=df_delta.reset_index()
pdf["ds"]=pdf["data"]
#pdf.tail(5)
def forecast(ds_forecast,column,days=7):
    ds_forecast["y"]=ds_forecast[column]
    pro_df=ds_forecast[["ds","y"]]
    my_model = Prophet()
    my_model.fit(pro_df)
    future_dates = my_model.make_future_dataframe(periods=days, freq='D')
    forecast = my_model.predict(future_dates)
    return my_model,forecast 

model,fc=forecast(pdf,"delta_deceduti")
model.plot(fc,uncertainty=True);
pdf_tmp=pdf[pdf["ds"]>'2020-03-21'].copy()
model,fc=forecast(pdf_tmp,"delta_deceduti")
model.plot(fc,uncertainty=True);
model,fc=forecast(pdf,"delta_dimessi_guariti")
model.plot(fc,uncertainty=True);
model,fc=forecast(pdf,"delta_terapia_intensiva")
model.plot(fc,uncertainty=True);
model,fc=forecast(pdf,"delta_totale_positivi")
model.plot(fc,uncertainty=True);
model,fc=forecast(pdf_tmp,"delta_totale_positivi")
model.plot(fc,uncertainty=True);
model,fc=forecast(pdf,"delta_totale_casi")
model.plot(fc,uncertainty=True);
model,fc=forecast(pdf_tmp,"delta_totale_casi")
model.plot(fc,uncertainty=True);
