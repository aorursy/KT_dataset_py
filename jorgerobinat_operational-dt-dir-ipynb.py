#Select station
station = "Cortegada" # stations ["Cortegada", "Coron"]


# Select output format
H_resolution = False # True 1.3 Km False 4 Km


#Select date and time
date_input = "2020-08-07" # date forecast format "yyyy-mm-dd"
 
hour = 9 # UTC from 0 to 23 
import warnings
warnings.filterwarnings("ignore")
!pip install simplekml
import simplekml
import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix ,classification_report 
from sklearn.model_selection import cross_val_score,cross_validate
import seaborn as sns
from sklearn import preprocessing



#@title select 
labels = ["NE","SE","SW","NW","VRB"]
threshold = 2 
show_graph = True 
delete_ten_minutes = False 


#load database

if station=="Coron":
  join=pd.read_csv("../input/wind-coron/coronD1res4K.csv")
else:
  join=pd.read_csv("../input/wind-coron/cortegadaD1res4K.csv")

table_columns=[]
table=[]
table_index=[]

#x_var to obtain table columns, table, table_index. No need "mod" variables
X_var=["dir_NE","dir_SE","dir_SW","dir_NW"]
for var_pred0 in X_var:
  var_obs="dir_o"
  join[var_obs+"_l"]=pd.cut(join[var_obs], bins = len(labels), labels = labels).astype(str)
  join[var_pred0+"_l"]=pd.cut(join[var_pred0],bins = len(labels),labels=labels).astype(str)
  join.loc[join['spd_o'] < threshold, [var_obs+"_l"]] = "VRB"      
  join.loc[join["mod_"+var_pred0[-2:]]< threshold,[var_pred0+"_l"]]="VRB"

  #results tables
  res_df=pd.DataFrame({"pred_var":join[var_pred0+"_l"],"obs_var":join[var_obs+"_l"]})
  table.append(pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,))
  table_columns.append(pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,normalize="columns"))
  table_index.append(pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,normalize="index")  )




from urllib.request import urlretrieve
from datetime import datetime, timedelta, date
from urllib.request import urlretrieve
import xarray as xr

#X_var 8 variables
if station=="Coron":
  filename_in ="../input/wind-coron/algorithm/coron/coronD1res4K_treedir.h5"
else:
  filename_in = "../input/wind-coron/algorithm/cortegada/cortegadaD1res4K_treedir.h5"




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



#label model results

interval=pd.cut(join[var_obs],4,retbins = True,)[0].cat.categories
correspondence={"(-0.36, 90.0]":"NE","(90.0, 180.0]":"SE","(180.0, 270.0]":"SW","(270.0, 360.0]":"NW"}
model["dir_NE_l"]=pd.cut(model["dir_NE"],bins = interval).astype(str).map(correspondence)
model.loc[model['mod_NE'] < threshold, ["dir_NE_l"]] = "VRB"  
model["dir_SE_l"]=pd.cut(model["dir_SE"],bins = interval).astype(str).map(correspondence)
model.loc[model['mod_SE'] < threshold, ["dir_SE_l"]] = "VRB"  
model["dir_SW_l"]=pd.cut(model["dir_SW"],bins = interval).astype(str).map(correspondence)
model.loc[model['mod_SW'] < threshold, ["dir_SW_l"]] = "VRB"  
model["dir_NW_l"]=pd.cut(model["dir_NW"],bins = interval).astype(str).map(correspondence)
model.loc[model['mod_NW'] < threshold, ["dir_NW_l"]] = "VRB"  

#load 

clf1 = pickle.load(open(filename_in, 'rb'))
#get Y

Y=join[var_obs+"_l"]

#independent variables. With mod variable
X_var = ['dir_NE', 'dir_SE','dir_NW', 'dir_SW',"mod_NE","mod_SE","mod_SW","mod_NW"]
X=join[X_var]


#we  scale and split


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

y_pred=clf1.predict(x_test)
y_pred_df=pd.DataFrame({"var_pred":y_pred},index=y_test.index)
table_columns1=pd.crosstab(y_test,y_pred_df["var_pred"], margins=True,normalize="columns")





#scale X_var. Same scale NN trained


model[var_obs+"_ML"]=clf1.predict(model[X_var])




#station results
try:
  station_r=True
  variables_station=["spd_o_corte","std_spd_o_corte","gust_spd_o_corte","dir_o_corte"]
  param=["param=81","param=10009","param=10003","param=82"]

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
  df_station["dir_o_corte_l"]=pd.cut(df_station["dir_o_corte"], bins = interval).astype(str).map(correspondence)  
  df_station.loc[df_station['spd_o_corte'] < threshold, ["dir_o_corte_l"]] = "VRB" 
except:
  station_r=False
  df_station=pd.DataFrame(index=model.index,columns=["dir_o_corte", "dir_o_corte_l"])  

#merge station with meteorological model and plot

final=pd.merge(model, df_station, left_index=True, right_index=True, how='outer')
if show_graph and station_r:
  g1=(final[['dir_NE',"dir_SE","dir_SW","dir_NW","dir_o_corte"]]).dropna().plot(title="wind dir",figsize=(9,5)).grid(True,which='both')
  

#reample observed data hourly and show all data about spd
pd.options.display.max_rows = 999

if delete_ten_minutes:
  final_s=final[["dir_NE","dir_NE_l","dir_SE","dir_SE_l","dir_SW","dir_SW_l",
                 "dir_NW","dir_NW_l","dir_o_ML","dir_o_corte",
                 "dir_o_corte_l"]].dropna()
else:
  final_s=final[["dir_NE","dir_NE_l","dir_SE","dir_SE_l","dir_SW","dir_SW_l",
                 "dir_NW","dir_NW_l","dir_o_ML","dir_o_corte","dir_o_corte_l"]]



"""***********************************"""


q_df=final[["dir_NE_l","dir_SE_l","dir_SW_l","dir_NW_l",var_obs+"_ML"]].dropna()
pd.set_option('max_colwidth', 2000)
quantum_metmod_NE=[]
quantum_metmod_SE=[]
quantum_metmod_SW=[]
quantum_metmod_NW=[]
quantum_ML=[]
tablenew=[]
for i in range (0,4):
  tablenew.append(table_columns[i].rename(mapper=str,axis=1))
for i in range(0, len(q_df.index)):
  quantum_metmod_NE.append(tablenew[0][q_df["dir_NE_l"][i]].map("{:.0%}".format))
  quantum_metmod_SE.append(tablenew[1][q_df["dir_SE_l"][i]].map("{:.0%}".format))
  quantum_metmod_SW.append(tablenew[2][q_df["dir_SW_l"][i]].map("{:.0%}".format))
  quantum_metmod_NW.append(tablenew[3][q_df["dir_NW_l"][i]].map("{:.0%}".format))
 
  quantum_ML.append(table_columns1[q_df[var_obs+"_ML"][i]].map("{:.0%}".format))
  
quantum_fi=pd.DataFrame({"NE":quantum_metmod_NE,"SE":quantum_metmod_SE,
                         "SW":quantum_metmod_SW,"NW":quantum_metmod_NW,
                         "ML":quantum_ML}, index=q_df.index)





variable_met = "dir"


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


  

df_n[variable_met]=(round(df_n[variable_met],-1)).astype(str)
kml = simplekml.Kml()
df_n.apply(lambda X: kml.newpoint(name=X[variable_met], coords=[( X["lon"],X["lat"])]) ,axis=1)

#add description tag
tag= "Wind direction\n"
  
#add Cortegada velocity and ML prediction
description=tag+quantum_fi.columns[4]+" "+str(quantum_fi.iloc[hour,4])[:-15]
string=final.index.strftime("%Y-%m-%d")[0]+" "+str(hour)+":00:00"

if station=="Cortegada":
  kml.newpoint(name=str(final['dir_o_corte_l'].loc[string]), description=description,coords=[(-8.7836,42.6255)]) 
else:
  kml.newpoint(name="Coron", description=description,coords=[(-8.8046,42.5801)]) 

#Add model stadistical results four corners
if station=="Cortegada":
  descriptionNE=tag+quantum_fi.columns[0]+" "+str(quantum_fi.iloc[hour,0])[:-15]
  kml.newpoint(name=str(final['dir_NE_l'].loc[string]),description=descriptionNE,coords=[(-8.7557,42.6446)])
  descriptionSE=tag+quantum_fi.columns[1]+" "+str(quantum_fi.iloc[hour,1])[:-15]
  kml.newpoint(name=str(final['dir_SE_l'].loc[string]),description=descriptionSE,coords=[(-8.7588,42.6090)])
  descriptionSW=tag+quantum_fi.columns[2]+" "+str(quantum_fi.iloc[hour,2])[:-15]
  kml.newpoint(name=str(final['dir_SW_l'].loc[string]),description=descriptionSW,coords=[(-8.8076,42.6115)])
  descriptionNW=tag+quantum_fi.columns[3]+" "+str(quantum_fi.iloc[hour,3])[:-15]
  kml.newpoint(name=str(final['dir_NW_l'].loc[string]),description=descriptionNW,coords=[(-8.8045,42.6469)])  
else:
  descriptionNE=tag+quantum_fi.columns[0]+" "+str(quantum_fi.iloc[hour,0])[:-15]
  kml.newpoint(name=str(final['dir_NE_l'].loc[string]),description=descriptionNE,coords=[(-8.7588,42.6080)])
  descriptionSE=tag+quantum_fi.columns[1]+" "+str(quantum_fi.iloc[hour,1])[:-15]
  kml.newpoint(name=str(final['dir_SE_l'].loc[string]),description=descriptionSE,coords=[(-8.7619,42.5729)])
  descriptionSW=tag+quantum_fi.columns[2]+" "+str(quantum_fi.iloc[hour,2])[:-15]
  kml.newpoint(name=str(final['dir_SW_l'].loc[string]),description=descriptionSW,coords=[(-8.8107,42.5752)])
  descriptionNW=tag+quantum_fi.columns[3]+" "+str(quantum_fi.iloc[hour,3])[:-15]
  kml.newpoint(name=str(final['dir_NW_l'].loc[string]),description=descriptionNW,coords=[(-8.8076,42.6108)])  

  
  
#save results

kml.save(today+"H"+str(hour)+r+variable_met+"_ML"+".kml")




final_s
quantum_fi