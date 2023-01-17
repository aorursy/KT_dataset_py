quantile0 =  5#@param ["2", "4", "10", "5"] {type:"raw", allow-input: true}

var_pred0 = "mod_corte" #@param ["mod_coron", "mod_corte", "spd_o_coron", "spd_o_corte", "wind_gust_corte", "wind_gust_coron"]

var_obs0 = "spd_o_corte" #@param ["spd_o_corte", "spd_o_coron", "gust_spd_o_corte", "gust_spd_o_coron"]

!pip install simplekml

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix ,classification_report 

from sklearn.model_selection import cross_val_score,cross_validate

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

from sklearn import tree

import graphviz

import pickle

import simplekml

import datetime

from datetime import timedelta

from urllib.request import urlretrieve

import urllib.request

import xarray as xr



def wind_speed_class_test(q,spd_pred,spd_obs,w_df):

    

    """

    Convert wind velocity in labels above a wind intensity forecast and compare

    wind direction observed plot results as a crosstabulation

    

    parameters:

    

    

    q=Number of quantiles

    spd_pred=column from the dataframe (w_df) where the wind is predicted;

    spd_obs:column from the dataframe (w_df) where the wind is observed; 

    w_df=dataframe where all columns are """

    

    

        

    #first q cut observed variable then cut predicted with the bins obtained at qcut

    w_df[spd_obs+"_l"]=pd.qcut(w_df[spd_obs], q, retbins = False,precision=1)

    interval=pd.qcut(w_df[spd_obs], q,retbins = True,precision=1)[0].cat.categories

    w_df[spd_pred+"_l"]=pd.cut(w_df[spd_pred],bins = interval)

    

    

    #plot results

    table=pd.crosstab(w_df[spd_obs+"_l"], w_df[spd_pred+"_l"], margins=True,)

    table_columns=pd.crosstab(w_df[spd_obs+"_l"], w_df[spd_pred+"_l"], margins=True,normalize="columns")

    table_index=pd.crosstab(w_df[spd_obs+"_l"], w_df[spd_pred+"_l"], margins=True,normalize="index")

    

    fig, axs = plt.subplots(3,figsize = (8,10))

    sns.heatmap(table,annot=True,ax=axs[0],cmap="YlGnBu",fmt='.0f',)

    sns.heatmap(table_columns,annot=True,ax=axs[1],cmap="YlGnBu",fmt='.0%')

    sns.heatmap(table_index,annot=True,ax=axs[2],cmap="YlGnBu",fmt=".0%")

    plt.show()



def class_report  (q,spd_pred,spd_obs,w_df):

    

    """

    Convert wind velocity in quantiles and compare

    wind velocity plot results: precision,recall, accuracy and f1 

    

    parameters:

    q=Number of quantiles

    spd_pred=column from the dataframe (w_df) where the wind is predicted;

    spd_obs:column from the dataframe (w_df) where the wind is observed; 

    w_df=dataframe where all columns are 

    

   """

    #first q cut observed variable then cut predicted with the bins obtained at qcut

    w_df[spd_obs+"_l"]=pd.qcut(w_df[spd_obs], q, retbins = False,precision=1).astype(str)

    interval=pd.qcut(w_df[spd_obs], q,retbins = True,precision=1)[0].cat.categories

    w_df[spd_pred+"_l"]=pd.cut(w_df[spd_pred],bins = interval).astype(str)

        

    

    #classification report

    #nan (=velocity predicted more than the hight velocity observed)

    #transform labels as string type

    

    print(classification_report(w_df[spd_obs+"_l"],w_df[spd_pred+"_l"]))

  









  



coron=pd.read_csv('/kaggle/input/wind-coron/coron_all.csv',parse_dates=["time"]).set_index("time")

cortegada=pd.read_csv('/kaggle/input/wind-coron/cortegada_all.csv',parse_dates=["time"]).set_index("time")

join = cortegada.join(coron, lsuffix='_corte', rsuffix='_coron').dropna()

class_report(quantile0,var_pred0,var_obs0,join)

wind_speed_class_test(quantile0,var_pred0,var_obs0,join)

#@title Load or train model. If load, same quantile and var_pred1

quantile1 =  5#@param ["2", "4", "10", "5"] {type:"raw", allow-input: true}

var_pred1 =  ['mod_coron', 'wind_gust_coron','mod_corte', 'wind_gust_corte'] #@param ["['mod_coron', 'wind_gust_coron','mod_corte', 'wind_gust_corte']", "['mod_coron', 'wind_gust_coron']", "['mod_corte', 'wind_gust_corte']", "['gust_spd_o_coron', 'spd_o_coron' ]", "['gust_spd_o_corte', 'spd_o_corte' ]"] {type:"raw"}

var_obs1 = "spd_o_corte" #@param ["spd_o_corte", "spd_o_coron", "gust_spd_o_corte", "gust_spd_o_coron", "std_spd_o_corte", "std_spd_o_coron"]

max_depth=5#@param ["2", "5", "10", "15"] {type:"raw", allow-input: true}

criterion = "entropy" #@param ["gini", "entropy"]

train= True #@param {type:"boolean"}

filename_in = "/kaggle/input/wind-coron/tree.h5" #@param ["/kaggle/input/wind-coron/tree.h5"] {type:"raw", allow-input: true}

filename_out = "tree2.h5" #@param ["tree2.h5""] {type:"raw", allow-input: true}





#cut in bins==wind quadrants exemple

Y=pd.qcut(join[var_obs1], quantile1, retbins = False,precision=1).astype(str)

labels=pd.qcut(join[var_obs1], quantile1,retbins = True,precision=1)[0].cat.categories.astype(str)

#independent variables. Also observed variables!! if you wish

X=join[var_pred1]





#we do not scale!!

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,)



#select classification

#train and save as tree2 or load tree

if train:

  clf1 = DecisionTreeClassifier(max_depth=max_depth,criterion=criterion).fit(x_train,y_train) 

  pickle.dump(clf1, open(filename_out, 'wb'))

else:

  #model loaded must have same X variables and q

  clf1 = pickle.load(open(filename_in, 'rb'))



y_pred=clf1.predict(x_test)



#plot results

print(classification_report(y_test,y_pred))

y_pred_df=pd.DataFrame({"var_pred":y_pred},index=y_test.index)



#plot results

table=pd.crosstab(y_test,y_pred_df["var_pred"], margins=True,)

table_columns1=pd.crosstab(y_test,y_pred_df["var_pred"], margins=True,normalize="columns")

table_index=pd.crosstab(y_test,y_pred_df["var_pred"], margins=True,normalize="index")





fig, axs = plt.subplots(3,figsize = (8,10))

sns.heatmap(table,annot=True,ax=axs[0],cmap="YlGnBu",fmt='.0f',)

sns.heatmap(table_columns1,annot=True,ax=axs[1],cmap="YlGnBu",fmt='.0%')

sns.heatmap(table_index,annot=True,ax=axs[2],cmap="YlGnBu",fmt=".0%")

plt.show()





print("Features importances:")

fi=["{:.0%}".format(x) for x in clf1.feature_importances_]

print(dict(zip(X.columns,fi )))





#cross validation



print ("***Accuracy score***")

print(cross_val_score(clf1, X, Y, cv=10,scoring="accuracy"))

print ("***F1_macro score***")

print(cross_val_score(clf1, X, Y, cv=10,scoring='f1_macro'))



#tree save file name: tree

dot_data = tree.export_graphviz(clf1, out_file=None, 

                                  feature_names=X.columns,  

                                  class_names=labels,  

                                  filled=True, rounded=True,  

                                  special_characters=True)  

graph = graphviz.Source(dot_data) 

graph.render("tree")

   
quantile2 =  10#@param ["2", "4", "10", "5"] {type:"raw", allow-input: true}

var_pred2 =  ['mod_coron', 'wind_gust_coron','mod_corte', 'wind_gust_corte'] #@param ["['mod_coron', 'wind_gust_coron','mod_corte', 'wind_gust_corte']", "['mod_coron', 'wind_gust_coron']", "['mod_corte', 'wind_gust_corte']", "['gust_spd_o_coron', 'spd_o_coron' ]", "['gust_spd_o_corte', 'spd_o_corte' ]"] {type:"raw"}

var_obs2 = var_obs1

train= True #@param {type:"boolean"}

filename_in = "/kaggle/input/wind-coron/rforest.h5" #@param ["/content/drive/My Drive/Colab Notebooks/wind_ria_arousa/rforest.h5"] {type:"raw", allow-input: true}

filename_out = "rforest2.h5" #@param ["rforest2.h5"] {type:"raw", allow-input: true}



#cut in bins==wind quadrants exemple

Y=pd.qcut(join[var_obs2], quantile2, retbins = False,precision=1).astype(str)

labels=pd.qcut(join[var_obs2], quantile2,retbins = True,precision=1)[0].cat.categories.astype(str)

#independent variables. Also observed variables!! if you wish

X=join[var_pred2]





#we do not scale!!

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,)



#select classification

if train:

  clf2 = RandomForestClassifier().fit(x_train,y_train) 

  pickle.dump(clf2, open(filename_out, 'wb'))

else:

  #model loaded must have same X variables and q

  clf2 = pickle.load(open(filename_in, 'rb'))







y_pred=clf2.predict(x_test)



#plot results

print(classification_report(y_test,y_pred))

y_pred_df=pd.DataFrame({"var_pred":y_pred},index=y_test.index)



#plot results

table=pd.crosstab(y_test,y_pred_df["var_pred"], margins=True,)

table_columns2=pd.crosstab(y_test,y_pred_df["var_pred"], margins=True,normalize="columns")

table_index=pd.crosstab(y_test,y_pred_df["var_pred"], margins=True,normalize="index")





fig, axs = plt.subplots(3,figsize = (8,10))

sns.heatmap(table,annot=True,ax=axs[0],cmap="YlGnBu",fmt='.0f',)

sns.heatmap(table_columns2,annot=True,ax=axs[1],cmap="YlGnBu",fmt='.0%')

sns.heatmap(table_index,annot=True,ax=axs[2],cmap="YlGnBu",fmt=".0%")

plt.show()



#@title Plot results

delete_ten_minutes = True #@param {type:"boolean"}

show_graph = True #@param {type:"boolean"}

date_input = '2020-06-11' #@param {type:"date"}





from datetime import datetime, timedelta, date

from urllib.request import urlretrieve

import xarray as xr



date_input=datetime.strptime(date_input,  '%Y-%m-%d')

#getting model variables



#creating the string_url

#analysis day= Yesterday. Time 00:00Z. 

datetime_str = (date_input-timedelta(days = 1)).strftime('%Y%m%d')



#day to forecast 1= D+1 , 2 =D+2 and so on 

forecast=1

dataframes=[]

date_anal = datetime.strptime(datetime_str,'%Y%m%d')

date_fore=(date_anal+timedelta(days=forecast)).strftime('%Y-%m-%d')



#Coron lat: 42.580 N  lon: 8.8047 W. Cortegada lat: 42.626 N  lon: 8.784 W

coordenates=["latitude=42.626&longitude=-8.784&","latitude=42.580&longitude=-8.8047&"]

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

model = dataframes[0].join(dataframes[1], lsuffix='_corte', rsuffix='_coron')



#model forecast bins and Machine learning forecast

interval=pd.qcut(join[var_obs0], quantile0,retbins = True,precision=1)[0].cat.categories

model[var_pred0+"_l"]=pd.cut(model[var_pred0],bins = interval).astype(str)

model[var_obs1+"_decisiontree"]=clf1.predict(model[var_pred1])

model[var_obs2+"_randomforest"]=clf2.predict(model[var_pred2])







#station results

variables_station=["spd_o_corte","std_spd_o_corte","gust_spd_o_corte"]

param=["param=81","param=10009","param=10003"]



head="http://www2.meteogalicia.gal/galego/observacion/plataformas/historicosAtxt/DatosHistoricosTaboas_dezminutalAFicheiro.asp?"



"""Cortegada platform:15001, Ribeira buoy:15005 warnings: wind intensity negatives!!"""

station="est=15001&"





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

  url3=head+station+parameters+dateday



  #decimal are comma ,!!

  df=pd.read_fwf(url3,skiprows=24,sep=" ",encoding='latin-1',decimal=',').dropna()

  df_station["datetime"]=df["DATA"]+" "+df['Unnamed: 2']

  df_station['datetime'] = pd.to_datetime(df_station['datetime'])

  df_station[var]=df['Valor'].astype(float)

df_station=df_station.set_index("datetime") 



#merge station with meteorological model and plot



final=pd.merge(model, df_station, left_index=True, right_index=True, how='outer')

if show_graph:

  g1=(final[['mod_corte',"mod_coron","spd_o_corte"]]*1.9438).dropna().plot(title="wind velocity KT",figsize=(9,5)).grid(True,which='both')

  g2=(final[['mod_corte',"mod_coron",]]*1.9438).dropna().plot(title="wind velocity KT",figsize=(9,5)).grid(True,which='both')

  g3=(final[['mod_corte',"mod_coron","spd_o_corte"]][1:]*1.9438).plot(title="wind velocity KT",figsize=(9,5)).grid(True,which='both')

  g4=(final[["wind_gust_corte"	,"wind_gust_coron","gust_spd_o_corte"]]*1.9438).plot(title="wind gust velocity KT",figsize=(9,5)).grid(True,which='both')

  g5=final[['dir_corte','dir_coron']].plot(title="Wind direction",figsize=(9,5)).grid(True,which='both')



#reample observed data hourly and show all data about spd

pd.options.display.max_rows = 999

final["observed_resample_hourly"]=final.spd_o_corte.resample("H").mean()



if delete_ten_minutes:

  final_s=final[["spd_o_corte",	"std_spd_o_corte","gust_spd_o_corte",

                 "observed_resample_hourly",var_pred0,var_pred0+"_l",

                 var_obs1+"_decisiontree",var_obs2+"_randomforest"]].dropna()

else:

  final_s=final[["spd_o_corte",	"std_spd_o_corte","gust_spd_o_corte",

                 "observed_resample_hourly",var_pred0,var_pred0+"_l",

                 var_obs1+"_decisiontree",var_obs2+"_randomforest"]]

final_s
#@title quantum results

q_df=final[[var_obs1+"_decisiontree",var_obs2+"_randomforest"]].dropna()

pd.set_option('max_colwidth', 2000)

quantum_randomforest=[]

quantum_decisiontree=[]

formatter="{'.0%'}".format

for i in range(0, len(q_df.index)):

  quantum_randomforest.append(table_columns2[q_df[var_obs2+"_randomforest"][i]].map("{:.0%}".format))

  quantum_decisiontree.append(table_columns1[q_df[var_obs1+"_decisiontree"][i]].map("{:.0%}".format))

  

quantum_fi=pd.DataFrame({var_obs1+"_decision tree":quantum_decisiontree,var_obs2+"_randomforest":quantum_randomforest}, index=q_df.index)

quantum_fi

#@title Select time forecast

hour = 12 #@param {type:"slider", min:0, max:23, step:1}

knots = True #@param {type:"boolean"}

celsius = False

H_resolution = True #@param {type:"boolean"}

variable_met = "wind_gust" #@param ["wind_gust", "mod", "temp", "prec", "dir"] {allow-input: true}





today=date_input

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

  df_n[variable_met]=round(df_n[variable_met]*1.94384,2).astype(int)

  

if variable_met=="temp" and celsius:

  df_n[variable_met]=(df_n[variable_met]-273.16).astype(int)

 

if variable_met=="dir":

   df_n[variable_met]= df_n[variable_met].astype(int)





df_n[variable_met]=df_n[variable_met].astype(str)

kml = simplekml.Kml()

df_n.apply(lambda X: kml.newpoint(name=X[variable_met], coords=[( X["lon"],X["lat"])]) ,axis=1)



#add Cortegada wind variables if variable_met mod or wind_gust

description="units m/s\n"+quantum_fi.columns[0]+" "+str(quantum_fi.iloc[hour,0])[:-15]+"\n"+quantum_fi.columns[1]+" "+str(quantum_fi.iloc[hour,1])[:-15]



if variable_met=="mod" or variable_met=="wind_gust":

  #add Cortegada velocity

  string=final.index.strftime("%Y-%m-%d")[0]+" "+str(hour)+":00:00"

  if var_obs1[-5:]=="coron":

    description="*"

  if knots:

    kml.newpoint(name=str(round(final['spd_o_corte'].loc[string]*1.9438,0)), description=description,coords=[(-8.7836,42.6255)]) 

  else:

    kml.newpoint(name=str(final['spd_o_corte'].loc[string]), description=description,coords=[(-8.7836,42.6255)]) 



  #add Cortegada gust

  if knots:

    kml.newpoint(name=str(round(final['gust_spd_o_corte'].loc[string]*1.9438,0)), description=description,coords=[(-8.7836,42.6255)]) 

  else:

    kml.newpoint(name=str(final['gust_spd_o_corte'].loc[string]), description=description,coords=[(-8.7836,42.6255)]) 

   #add Coron

  description="units m/s\n"+quantum_fi.columns[0]+" "+str(quantum_fi.iloc[hour,0])[:-15]+"\n"+quantum_fi.columns[1]+" "+str(quantum_fi.iloc[hour,1])[:-15]

  if var_obs1[-5:]=="corte":

    description="*"

  kml.newpoint(name="Coron",description=description,coords=[(-8.8046,42.5801)])   



#save results  

kml.save(today+"H"+str(hour)+r+variable_met+".kml")