!pip install simplekml
from urllib.request import urlretrieve
import urllib.request
import xarray as xr
import numpy as np
import pandas as pd
import simplekml
import datetime
from datetime import timedelta
#@title Select time forecast
hour = 14 #@param {type:"slider", min:0, max:23, step:1}
knots = True #@param {type:"boolean"}
celsius = True #@param {type:"boolean"}
H_resolution = True #@param {type:"boolean"}
variable_met = "mod" #@param ["wind_gust", "mod", "temp", "prec", "dir"] {allow-input: true}


today=datetime.datetime.now()
yesterday=today+timedelta(days=-1)
today=today.strftime("%Y-%m-%d")
yesterday=yesterday.strftime("%Y%m%d")


url1="http://mandeo.meteogalicia.es/thredds/ncss/wrf_2d_04km/fmrc/files/"+yesterday+"/wrf_arw_det_history_d03_"+yesterday+"_0000.nc4?var=lat&var=lon&var="+variable_met+"&north=42.650&west=-9.00&east=-8.75&south=42.450&disableProjSubset=on&horizStride=1&time_start="+today+"T"+str(hour)+"%3A00%3A00Z&time_end="+today+"T"+str(hour)+"%3A00%3A00Z&timeStride=1&accept=netcdf"
url2="http://mandeo.meteogalicia.es/thredds/ncss/wrf_1km_baixas/fmrc/files/"+yesterday+"/wrf_arw_det1km_history_d05_"+yesterday+"_0000.nc4?var=lat&var=lon&var="+variable_met+"&north=42.650&west=-9.00&east=-8.75&south=42.450&disableLLSubset=on&disableProjSubset=on&horizStride=1&time_start="+today+"T"+str(hour)+"%3A00%3A00Z&time_end="+today+"T"+str(hour)+"%3A00%3A00Z&timeStride=1&accept=netcdf"
if H_resolution:
  url=url2
  r="HI_"
else:
  url=url1
  r="LO_"


urlretrieve(url,"model")
df=xr.open_dataset("model").to_dataframe()
df_n=pd.DataFrame(df[["lat","lon",variable_met]].values,columns=df[["lat","lon",variable_met]].columns)
if knots and (variable_met=="mod" or variable_met=="wind_gust"):
  df_n[variable_met]=round(df_n[variable_met]*1.94384,0).astype(int)
if variable_met=="temp" and celsius:
  df_n[variable_met]=(df_n[variable_met]-273.16).astype(int)
if variable_met!="prec":
   df_n[variable_met]= df_n[variable_met].astype(int)


df_n[variable_met]=df_n[variable_met].astype(str)
kml = simplekml.Kml()
df_n.apply(lambda X: kml.newpoint(name=X[variable_met], coords=[( X["lon"],X["lat"])]) ,axis=1)
kml.save(today+"H"+str(hour)+r+variable_met+".kml")