from datetime import timedelta

from urllib.request import urlretrieve

import urllib.request

import xarray as xr

import numpy as np

import pandas as pd

head="http://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/" # url head

res="d01" # Grid resolution: d03 grid 4 Km, d02 grid 12Km and d01 grid 36Km

forecast=1 # forecast=1 D+1 , forecast=2 D+2, forecast 3 D+3

coordenates="latitude=42.23&longitude=-8.63" # North and East + and South and West -



var1="var=HGT500&var=HGT850&var=HGTlev1&var=HGTlev2&var=HGTlev3&var=T500&var=T850&var=cape&"

var2="var=cfh&var=cfl&var=cfm&var=cft&var=cin&var=conv_prec&var=dir&var=lhflx&"

var3="var=lwflx&var=lwm&var=meteograms&var=mod&var=mslp&var=pbl_height&var=prec&"

var4="var=rh&var=shflx&var=snow_prec&var=snowlevel&var=sst&var=swflx&var=temp&"

var5="var=topo&var=u&var=ulev1&var=ulev2&var=ulev3&var=v&var=visibility&"

var6="var=vlev1&var=vlev2&var=vlev3&var=weasd&var=wind_gust&"

var_tot=var1+var2+var3+var4+var5+var6



df_sum=pd.DataFrame({'A' : []}) # start data frame null

#select start and end date format mm/dd/yyyy!!!



for date in pd.date_range(start='12/25/2018', end='12/26/2019'):

  print("date:",date)

  time_fore=(date+timedelta(days=forecast)).strftime('%Y-%m-%d')

  url=head+res+date.strftime('/%Y/%m/')+"wrf_arw_det_history_"+res+date.strftime("_%Y%m%d_0000.nc4?")+var_tot+coordenates+"&time_start="+time_fore+"T00%3A00%3A00Z&time_end="+time_fore+"T23%3A00%3A00Z&accept=netcdf"

  try:

    urlretrieve(url,"model")

    df=xr.open_dataset("model").to_dataframe().set_index("time").loc[:, 'HGT500':'wind_gust']

    df_sum=pd.concat([df_sum,df],sort=False)

  except:

    print(date,"failed")

    

df_sum=df_sum.drop(['A'], axis=1)

# select filename LEVX Vigo airport

file_name="LEVX"+"R36KM"+"D"+str(forecast)

df_sum.to_csv(file_name)

df_sum