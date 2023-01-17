#Select station



station = "Cortegada" # stations ["Cortegada", "Coron"]





# Select output format

 

quantile =  5 # select quintiles, deciles ... (integer input)



beaufort = True # True Beaufort False quintiles



knots = True # True knots False m/s



H_resolution = False # True 1.3 Km False 4 Km





#Select date and time

 

date_input = "2020-08-17" # date forecast format "yyyy-mm-dd" if today=D you can select from D-10 to D+1

 

hour = 3 # UTC from 0 to 23 

import warnings

warnings.filterwarnings("ignore")

!pip install simplekml

import simplekml

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix ,classification_report 

from sklearn.model_selection import cross_val_score,cross_validate

import seaborn as sns

from sklearn import preprocessing

from tensorflow.keras.models import load_model

import tensorflow as tf

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler





#@title select.



#X_var = ['mod_NE', 'mod_SE','mod_NW', 'mod_SW',"wind_gust_NE","wind_gust_SE" ,"wind_gust_SW","wind_gust_NW"] #@param ["['mod_NE', 'mod_SE','mod_NW', 'mod_SW',\"wind_gust_NE\",\"wind_gust_SE\" ,\"wind_gust_SW\",\"wind_gust_NW\"]", "['mod_NE', 'mod_SE','mod_NW', 'mod_SW']"] {type:"raw", allow-input: true}



delete_ten_minutes = False

show_graph = True 

X_var = ['mod_NE', 'mod_SE','mod_SW', 'mod_NW',"wind_gust_NE","wind_gust_SE" ,"wind_gust_SW","wind_gust_NW"]

if station=="Coron":

  join=pd.read_csv("../input/wind-coron/coronD1res4K.csv")

else:

  join=pd.read_csv("../input/wind-coron/cortegadaD1res4K.csv")

table_columns=[]

table=[]

table_index=[]



for var_pred0 in X_var:

  var_obs="spd_o"

  if beaufort:



    #first cut observed variable in Beaufort intervals

    bins_b = pd.IntervalIndex.from_tuples([(-1, 0.5), (.5, 1.5), (1.5, 3.3),(3.3,5.5),

                                     (5.5,8),(8,10.7),(10.7,13.8),(13.8,17.1),

                                     (17.1,20.7),(20.7,24.4),(24.4,28.4),(28.4,32.6),(32.6,60)])

    join[var_obs+"_l"]=pd.cut(join[var_obs], bins=bins_b).astype(str)

    join[var_pred0+"_l"]=pd.cut(join[var_pred0],bins = bins_b).astype(str)



    #transform to Beaufort scale

    bins_b=bins_b.astype(str)

    labels=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12"]

    join[var_obs+"_l"]=join[var_obs+"_l"].map({a:b for a,b in zip(bins_b,labels)})

    join[var_pred0+"_l"]=join[var_pred0+"_l"].map({a:b for a,b in zip(bins_b,labels)})



  else:

    #first q cut observed variable then cut predicted with the bins obtained at qcut

    join[var_obs+"_l"]=pd.qcut(join[var_obs], quantile, retbins = False,precision=1).astype(str)

    interval=pd.qcut(join[var_obs], quantile,retbins = True,precision=1)[0].cat.categories

    join[var_pred0+"_l"]=pd.cut(join[var_pred0],bins = interval).astype(str)

        

 



  #results tables

  res_df=pd.DataFrame({"pred_var":join[var_pred0+"_l"],"obs_var":join[var_obs+"_l"]})

  table.append(pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,))

  table_columns.append(pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,normalize="columns"))

  table_index.append(pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,normalize="index")  )













#@title Operational results.





from urllib.request import urlretrieve

from datetime import datetime, timedelta, date

from urllib.request import urlretrieve

import xarray as xr



#X_var 8 variables

if station=="Coron":

  if beaufort:

    filename_in = "../input/wind-coron/algorithm/coron/coronD1res4K_NN_b.h5"

  else:

    filename_in = "../input/wind-coron/algorithm/coron/coronD1res4K_NN_q.h5"

else:

  if beaufort:

    filename_in = "../input/wind-coron/algorithm/cortegada/cortegadaD1res4K_NN_b.h5"

  else:

    filename_in = "../input/wind-coron/algorithm/cortegada/cortegadaD1res4K_NN_q.h5"

date_input=datetime.strptime(date_input,  '%Y-%m-%d')

np.set_printoptions(formatter={'float_kind':"{'.0%'}".format})



#getting model variables



#creating the string_url

#analysis day= Yesterday. Time 00:00Z. 

datetime_str = (date_input-timedelta(days = 1)).strftime('%Y%m%d')



#day to forecast 1= D+1 , 2 =D+2 and so on 

forecast=1

dataframes=[]

date_anal = datetime.strptime(datetime_str,'%Y%m%d')

date_fore=(date_anal+timedelta(days=forecast)).strftime('%Y-%m-%d')



# points NE,SE,SW,Nw

if station=="Coron":

  coordenates=["latitude=42.6088&longitude=-8.7588&","latitude=42.5729&longitude=-8.7619&"

,"latitude=42.5752&longitude=-8.8107&","latitude=42.6110&longitude=-8.8076&"]

else:

  coordenates=["latitude=42.6446&longitude=-8.7557&","latitude=42.6088&longitude=-8.7588&"

,"latitude=42.6110&longitude=-8.8076&","latitude=42.6469&longitude=-8.8045&"]



#variables string type to perform url. The same variables as model (AI)



dataframes=[]

for coordenate in coordenates:

  head="http://mandeo.meteogalicia.es/thredds/ncss/wrf_2d_04km/fmrc/files/"

  text1="/wrf_arw_det_history_d03_"+datetime_str+"_0000.nc4?"

  met_var="var=dir&var=mod&var=wind_gust&"

  scope1="time_start="+date_fore+"T00%3A00%3A00Z&"

  scope2="time_end="+date_fore+"T23%3A00%3A00Z&accept=netcdf"

  #add all the string variables

  url=head+datetime_str+text1+met_var+coordenate+scope1+scope2

  #load the actual model from Meteogalicia database and transform as pandas dataframe

  urlretrieve(url,"model")

  dataframes.append(xr.open_dataset("model").to_dataframe().set_index("time").loc[:, 'dir':])

E = dataframes[0].join(dataframes[1], lsuffix='_NE', rsuffix='_SE')

W = dataframes[2].join(dataframes[3], lsuffix='_SW', rsuffix='_NW')

model=E.join(W)

if beaufort:

  #model forecast bins and Machine learning forecast

  bins_b = pd.IntervalIndex.from_tuples([(-1, 0.5), (.5, 1.5), (1.5, 3.3),(3.3,5.5),

                                     (5.5,8),(8,10.7),(10.7,13.8),(13.8,17.1),

                                     (17.1,20.7),(20.7,24.4),(24.4,28.4),(28.4,32.6),(32.6,60)])

  labels=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12"]

  model["mod_NE_l"]=pd.cut(model["mod_NE"],bins = bins_b).map({a:b for a,b in zip(bins_b,labels)})

  model["mod_SE_l"]=pd.cut(model["mod_SE"],bins = bins_b).map({a:b for a,b in zip(bins_b,labels)})

  model["mod_SW_l"]=pd.cut(model["mod_SW"],bins = bins_b).map({a:b for a,b in zip(bins_b,labels)})

  model["mod_NW_l"]=pd.cut(model["mod_NW"],bins = bins_b).map({a:b for a,b in zip(bins_b,labels)})

else:

  interval=pd.qcut(join[var_obs], quantile,retbins = True,precision=1)[0].cat.categories

  model["mod_NE_l"]=pd.cut(model["mod_NE"],bins = interval).astype(str)

  model["mod_SE_l"]=pd.cut(model["mod_SE"],bins = interval).astype(str)

  model["mod_SW_l"]=pd.cut(model["mod_SW"],bins = interval).astype(str)

  model["mod_NW_l"]=pd.cut(model["mod_NW"],bins = interval).astype(str)



#load 

mlp = tf.keras.models.load_model(filename_in)



#get table_columnsneural



if beaufort:

  Y=join[var_obs+"_l"]

else:

  Y=pd.qcut(join[var_obs], quantile, retbins = False,precision=1).astype(str)

  labels=pd.qcut(join[var_obs], quantile,retbins = True,precision=1)[0].cat.categories.astype(str)





#transform bins_label to label binary array



lb = preprocessing.LabelBinarizer()

lb.fit(labels)

Y=lb.transform(Y)

scaler=MinMaxScaler().fit(join[X_var])



#independent variables. 

X=scaler.transform(join[X_var])





#we  scale and split





x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state=42)



y_pred=mlp.predict(x_test)





#transform bynary array to label scale 



y_pred=lb.inverse_transform(y_pred)

y_test=lb.inverse_transform(y_test)









#plot results

res_df=pd.DataFrame({"pred_var":y_pred,"obs_var":y_test})

table=pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,)

table_columnsneural=pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,normalize="columns")

table_index=pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,normalize="index")







#from array to labels

lb = preprocessing.LabelBinarizer()

if beaufort:

  labels=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12"]

else:

  labels=interval.astype(str)





lb.fit(labels)



#scale X_var. Same scale NN trained



scaler=MinMaxScaler().fit(join[X_var])

model[var_obs+"_NN"]=lb.inverse_transform(mlp.predict(scaler.transform(model[X_var])))









#station results

try:

  station_r=True

  variables_station=["spd_o_corte","std_spd_o_corte","gust_spd_o_corte"]

  param=["param=81","param=10009","param=10003"]



  head="http://www2.meteogalicia.gal/galego/observacion/plataformas/historicosAtxt/DatosHistoricosTaboas_dezminutalAFicheiro.asp?"



  """Cortegada platform:15001, Ribeira buoy:15005 warnings: wind intensity negatives!!"""

  station_n="est=15001&"





  dateday="&data1="+date_input.strftime("%d/%m/%Y")+"&data2="+(date_input+timedelta(days = 1)).strftime("%d/%m/%Y")



  """param=83 (air temperature C) ,10018 (dew temperature C),86 (humidity%)

  ,81(wind speed m/s),10003 (wind gust m/s),10009 (std wind speed m/s)

  ,82 (wind direction degrees),10010 (std wind direction degrees),

  10015 (gust direcction degree),20003 (temperature sea surface C),20005 (salinity),

  20004 (conductivity mS/cm),20017 (density anomaly surface kg/m^3),20019 (deep sea temperature degrees)

  ,20018 (deep sea salinity),20022 (deep sea conductivity mS/cm),20021 (density anomaly deep sea kg/m^3),

  20020 (Presure water column db),20804 (East current compound cm/s) ,20803 (North current compound cm/s)"""



  df_station=pd.DataFrame()

  for parameters, var in zip(param,variables_station):

    url3=head+station_n+parameters+dateday



    #decimal are comma ,!!

    df=pd.read_fwf(url3,skiprows=24,sep=" ",encoding='latin-1',decimal=',').dropna()

    df_station["datetime"]=df["DATA"]+" "+df['Unnamed: 2']

    df_station['datetime'] = pd.to_datetime(df_station['datetime'])

    df_station[var]=df['Valor'].astype(float)

  df_station=df_station.set_index("datetime") 



  if beaufort:

    df_station["spd_o_corte_l"]=pd.cut(df_station["spd_o_corte"],bins = bins_b).map({a:b for a,b in zip(bins_b,labels)})

  else:

    df_station["spd_o_corte_l"]=pd.cut(df_station["spd_o_corte"],bins=interval).astype(str)  

  df_station["observed_resample_hourly"]=df_station.spd_o_corte.resample("H").mean()

except:

  station_r=False

  df_station=pd.DataFrame(index=model.index,columns=['spd_o_corte', 

                                                     'std_spd_o_corte', 

                                                     'gust_spd_o_corte',

                                                     'spd_o_corte_l',

                                                     "observed_resample_hourly"])







#merge station with meteorological model and plot



final=pd.merge(model, df_station, left_index=True, right_index=True, how='outer')

if show_graph and station_r:

  g1=(final[['mod_NE',"mod_SE","mod_SW","mod_NW","spd_o_corte"]]*1.9438).dropna().plot(title="wind velocity KT",figsize=(9,5)).grid(True,which='both')

  



#reample observed data hourly and show all data about spd

pd.options.display.max_rows = 999



if delete_ten_minutes:

  final_s=final[["mod_NE","mod_NE_l","mod_SE","mod_SE_l","mod_SW","mod_SW_l",

                 "mod_NW","mod_NW_l","spd_o_NN","spd_o_corte",

                 "spd_o_corte_l","gust_spd_o_corte","observed_resample_hourly"]].dropna()

else:

  final_s=final[["mod_NE","mod_NE_l","mod_SE","mod_SE_l","mod_SW","mod_SW_l",

                 "mod_NW","mod_NW_l","spd_o_NN","spd_o_corte","spd_o_corte_l",

                 "gust_spd_o_corte","observed_resample_hourly"]]







"""***********************************"""





q_df=final[["mod_NE_l","mod_SE_l","mod_SW_l","mod_NW_l",var_obs+"_NN"]].dropna()

pd.set_option('max_colwidth', 2000)

quantum_metmod_NE=[]

quantum_metmod_SE=[]

quantum_metmod_SW=[]

quantum_metmod_NW=[]

quantum_NN=[]

table=[]

for i in range (0,4):

  table.append(table_columns[i].rename(mapper=str,axis=1))

for i in range(0, len(q_df.index)):

  quantum_metmod_NE.append(table[0][q_df["mod_NE_l"][i]].map("{:.0%}".format))

  quantum_metmod_SE.append(table[1][q_df["mod_SE_l"][i]].map("{:.0%}".format))

  quantum_metmod_SW.append(table[2][q_df["mod_SW_l"][i]].map("{:.0%}".format))

  quantum_metmod_NW.append(table[3][q_df["mod_NW_l"][i]].map("{:.0%}".format))

 

  quantum_NN.append(table_columnsneural[q_df[var_obs+"_NN"][i]].map("{:.0%}".format))

  

quantum_fi=pd.DataFrame({"NE":quantum_metmod_NE,"SE":quantum_metmod_SE,

                         "SW":quantum_metmod_SW,"NW":quantum_metmod_NW,

                         "NN":quantum_NN}, index=q_df.index)











variable_met = "mod"





today=date_input

yesterday=today+timedelta(days=-1)

today=today.strftime("%Y-%m-%d")

yesterday=yesterday.strftime("%Y%m%d")





url1="http://mandeo.meteogalicia.es/thredds/ncss/wrf_2d_04km/fmrc/files/"+yesterday+"/wrf_arw_det_history_d03_"+yesterday+"_0000.nc4?var=lat&var=lon&var="+variable_met+"&north=42.68&west=-9.00&east=-8.65&south=42.250&disableProjSubset=on&horizStride=1&time_start="+today+"T"+str(hour)+"%3A00%3A00Z&time_end="+today+"T"+str(hour)+"%3A00%3A00Z&timeStride=1&accept=netcdf"

url2="http://mandeo.meteogalicia.es/thredds/ncss/wrf_1km_baixas/fmrc/files/"+yesterday+"/wrf_arw_det1km_history_d05_"+yesterday+"_0000.nc4?var=lat&var=lon&var="+variable_met+"&north=42.68&west=-9.00&east=-8.65&south=42.250&disableLLSubset=on&disableProjSubset=on&horizStride=1&time_start="+today+"T"+str(hour)+"%3A00%3A00Z&time_end="+today+"T"+str(hour)+"%3A00%3A00Z&timeStride=1&accept=netcdf"

if H_resolution:

  url=url2

  r="HI_"

else:

  url=url1

  r="LO_"





urlretrieve(url,"model")

df=xr.open_dataset("model").to_dataframe()

df_n=pd.DataFrame(df[["lat","lon",variable_met]].values,columns=df[["lat","lon",variable_met]].columns)



if knots:

  df_n[variable_met]=round(df_n[variable_met]*1.94384,2).astype(int)

  



df_n[variable_met]=df_n[variable_met].astype(str)

kml = simplekml.Kml()

df_n.apply(lambda X: kml.newpoint(name=X[variable_met], coords=[( X["lon"],X["lat"])]) ,axis=1)



#add description tag

if beaufort:

  tag= "Beaufort Scale\n"

else:

  tag="quintile\n"  

#add Cortegada velocity and ML prediction

description=tag+quantum_fi.columns[4]+" "+str(quantum_fi.iloc[hour,4])[:-15]

string=final.index.strftime("%Y-%m-%d")[0]+" "+str(hour)+":00:00"

if station=="Cortegada":

  kml.newpoint(name=str(final['spd_o_corte_l'].loc[string]), description=description,coords=[(-8.7836,42.6255)]) 

else:

  kml.newpoint(name="Coron", description=description,coords=[(-8.8046,42.5801)]) 



#Add model stadistical results four corners

if station=="Cortegada":

  descriptionNE=tag+quantum_fi.columns[0]+" "+str(quantum_fi.iloc[hour,0])[:-15]

  kml.newpoint(name=str(final['mod_NE_l'].loc[string]),description=descriptionNE,coords=[(-8.7557,42.6446)])

  descriptionSE=tag+quantum_fi.columns[1]+" "+str(quantum_fi.iloc[hour,1])[:-15]

  kml.newpoint(name=str(final['mod_SE_l'].loc[string]),description=descriptionSE,coords=[(-8.7588,42.6090)])

  descriptionSW=tag+quantum_fi.columns[2]+" "+str(quantum_fi.iloc[hour,2])[:-15]

  kml.newpoint(name=str(final['mod_SW_l'].loc[string]),description=descriptionSW,coords=[(-8.8076,42.6115)])

  descriptionNW=tag+quantum_fi.columns[3]+" "+str(quantum_fi.iloc[hour,3])[:-15]

  kml.newpoint(name=str(final['mod_NW_l'].loc[string]),description=descriptionNW,coords=[(-8.8045,42.6469)])  

else:

  descriptionNE=tag+quantum_fi.columns[0]+" "+str(quantum_fi.iloc[hour,0])[:-15]

  kml.newpoint(name=str(final['mod_NE_l'].loc[string]),description=descriptionNE,coords=[(-8.7588,42.6080)])

  descriptionSE=tag+quantum_fi.columns[1]+" "+str(quantum_fi.iloc[hour,1])[:-15]

  kml.newpoint(name=str(final['mod_SE_l'].loc[string]),description=descriptionSE,coords=[(-8.7619,42.5729)])

  descriptionSW=tag+quantum_fi.columns[2]+" "+str(quantum_fi.iloc[hour,2])[:-15]

  kml.newpoint(name=str(final['mod_SW_l'].loc[string]),description=descriptionSW,coords=[(-8.8107,42.5752)])

  descriptionNW=tag+quantum_fi.columns[3]+" "+str(quantum_fi.iloc[hour,3])[:-15]

  kml.newpoint(name=str(final['mod_NW_l'].loc[string]),description=descriptionNW,coords=[(-8.8076,42.6108)])  



#save results

if beaufort:

  kml.save("beaufort_"+today+"H"+str(hour)+r+variable_met+"_NN"+".kml")

else:

  kml.save("quintile_"+today+"H"+str(hour)+r+variable_met+"_NN"+".kml")







final_s
quantum_fi