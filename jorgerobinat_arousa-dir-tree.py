#@title Choose parameters and variables



threshold = 2  #@param {type: "slider", min: 0, max: 10}

var_threshold = "mod_corte" #@param ["mod_corte", "mod_coron", "spd_o_corte", "spd_o_corte"]

dir_pred0 = "dir_coron" #@param ["dir_corte", "dir_coron", "dir_o_corte", "dir_o_coron"]

dir_obs = "dir_o_coron" #@param ["dir_o_corte", "dir_o_coron"]

labels = ["NE","SE","SW","NW"] #@param ["[\"NE\",\"SE\",\"SW\",\"NW\"]", "[\"1\",\"2\"]", "[\"1\",\"2\",\"3\",\"4\",\"5\",\"6\"]", "[\"1\",\"2\",\"3\",\"4\",\"5\",\"6\",\"7\",\"8\",\"9\",\"10\"]"] {type:"raw", allow-input: true}



!pip install simplekml

import simplekml

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



def wind_direction_class_test(threshold,var_threshold,labels,dir_pred,dir_obs,w_df):

    

    """

    Convert wind direction in labels above a wind intensity forecast and compare

    wind direction observed plot results as a crosstabulation

    

    parameters:

    

    threshold=wind intensity thrsehold;

    var_threshold=variable wind intensity observed or forecasted same or different 

    location (column w_df);

    labels= to cut the continuity of wind direction example quadrants like NE or SE;

    dir_pred=column from the dataframe (w_df) where the wind is predicted;

    dir_obs:column from the dataframe (w_df) where the wind is observed; 

    w_df=dataframe where all columns are """

    

    

    #filter only data above a threshold

    mask=w_df[w_df[var_threshold]>=threshold].dropna()

    

    #cut in bins==wind quadrants exemple

    mask[dir_pred+"_l"]=pd.cut(x = mask[dir_pred], bins = len(labels), labels = labels)

    mask[dir_obs+"_l"]=pd.cut(x = mask[dir_obs], bins = len(labels), labels = labels)

    

    #plot results

    table=pd.crosstab(mask[dir_obs+"_l"], mask[dir_pred+"_l"], margins=True,)

    table_columns=pd.crosstab(mask[dir_obs+"_l"], mask[dir_pred+"_l"], margins=True,normalize="columns")

    table_index=pd.crosstab(mask[dir_obs+"_l"], mask[dir_pred+"_l"], margins=True,normalize="index")

    

    fig, axs = plt.subplots(3,figsize = (8,10))

    sns.heatmap(table,annot=True,ax=axs[0],cmap="YlGnBu",fmt='.0f',)

    sns.heatmap(table_columns,annot=True,ax=axs[1],cmap="YlGnBu",fmt='.0%')

    sns.heatmap(table_index,annot=True,ax=axs[2],cmap="YlGnBu",fmt=".0%")

    plt.show()



def class_report  (threshold,var_threshold,labels,dir_pred,dir_obs,w_df):

    

    """

    Convert wind direction in labels above a wind intensity forecast and compare

    wind direction plot results: precision,recall, accuracy and f1 

    

    parameters:

    

    threshold=wind intensity thrsehold;

    var_threshold=variable wind intensity observed or forecasted same or different 

    location (column w_df);

    labels= to cut the continuity of wind direction example quadrants like NE or SE;

    dir_pred=column from the dataframe (w_df) where the wind is predicted;

    dir_obs:column from the dataframe (w_df) where the wind is observed; 

    w_df=dataframe where all columns are """

    

    #filter only data above a threshold

    mask=w_df[w_df[var_threshold]>=threshold].dropna()

    

    #cut in bins==wind quadrants exemple

    mask[dir_pred+"_l"]=pd.cut(x = mask[dir_pred], bins = len(labels), labels = labels)

    mask[dir_obs+"_l"]=pd.cut(x = mask[dir_obs], bins = len(labels), labels = labels)

    

    #plot results

    print(classification_report(mask[dir_obs+"_l"],mask[dir_pred+"_l"]))

    











coron=pd.read_csv('/kaggle/input/wind-coron/coron_all.csv',parse_dates=["time"]).set_index("time")

cortegada=pd.read_csv('/kaggle/input/wind-coron/cortegada_all.csv',parse_dates=["time"]).set_index("time")

join = cortegada.join(coron, lsuffix='_corte', rsuffix='_coron').dropna()



w_df=join

class_report  (threshold,var_threshold,labels,dir_pred0,dir_obs,w_df)

wind_direction_class_test(threshold,var_threshold,labels,dir_pred0,dir_obs,w_df)
#@title Choose parameters and variables: Wind direction



threshold = 2  #@param {type: "slider", min: 0, max: 10}

var_threshold = "mod_corte" #@param ["mod_corte", "mod_coron", "spd_o_corte", "spd_o_corte"]

dir_pred1 = ['dir_coron', 'mod_coron', 'wind_gust_coron','dir_corte', 'mod_corte', 'wind_gust_corte',] #@param ["['dir_coron', 'mod_coron', 'wind_gust_coron','dir_corte', 'mod_corte', 'wind_gust_corte',\"dir_o_corte\"]", "['dir_coron', 'dir_corte', \"dir_o_corte\"]", "['dir_coron', 'dir_corte', \"dir_o_coron\"]", "['dir_coron', 'mod_coron', 'wind_gust_coron','dir_corte', 'mod_corte', 'wind_gust_corte',]", "[\"mod_corte\",\"mod_coron\"]", "[\"mod_corte\",\"mod_coron\",\"dir_corte\"]"] {type:"raw"}



labels = ["NE","SE","SW","NW"] #@param ["[\"NE\",\"SE\",\"SW\",\"NW\"]", "[\"1\",\"2\"]", "[\"1\",\"2\",\"3\",\"4\",\"5\",\"6\"]", "[\"1\",\"2\",\"3\",\"4\",\"5\",\"6\",\"7\",\"8\",\"9\",\"10\"]"] {type:"raw"}

max_depth = 10 #@param ["2", "5", "10", "15"] {type:"raw", allow-input: true}

criterion = "entropy" #@param ["gini", "entropy"]



#filter only data above a threshold

mask=join[join[var_threshold]>=threshold].dropna()

#cut in bins==wind quadrants exemple

Y=pd.cut(x = mask[dir_obs], bins = len(labels), labels = labels)



#independent variables. Also observed variables!! if you wish

X=mask[dir_pred1]





#we do not scale!!

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,)



#select classification

clf1 = DecisionTreeClassifier(max_depth=max_depth,criterion=criterion).fit(x_train,y_train) 

y_pred=clf1.predict(x_test)



#plot results

print(classification_report(y_test,y_pred))

y_pred_df=pd.DataFrame({"dir_pred":y_pred},index=y_test.index)



#plot results

table=pd.crosstab(y_test,y_pred_df["dir_pred"], margins=True,)

table_columns1=pd.crosstab(y_test,y_pred_df["dir_pred"], margins=True,normalize="columns")

table_index=pd.crosstab(y_test,y_pred_df["dir_pred"], margins=True,normalize="index")





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



#tree plot

dot_data = tree.export_graphviz(clf1, out_file=None, 

                     feature_names=X.columns,  

                     class_names=labels,  

                     filled=True, rounded=True,  

                     special_characters=True)  

graph = graphviz.Source(dot_data)  

graph.render("tree")  
#@title Choose parameters and variables: Wind direction



threshold = 2  #@param {type: "slider", min: 0, max: 10}

var_threshold = "mod_corte" #@param ["mod_corte", "mod_coron", "spd_o_corte", "spd_o_corte"]

dir_pred2 = ['dir_coron', 'mod_coron', 'wind_gust_coron','dir_corte', 'mod_corte', 'wind_gust_corte',] #@param ["['dir_coron', 'mod_coron', 'wind_gust_coron','dir_corte', 'mod_corte', 'wind_gust_corte',\"dir_o_corte\"]", "['dir_coron', 'dir_corte', \"dir_o_corte\"]", "['dir_coron', 'dir_corte', \"dir_o_coron\"]", "['dir_coron', 'mod_coron', 'wind_gust_coron','dir_corte', 'mod_corte', 'wind_gust_corte',]", "[\"mod_corte\",\"mod_coron\"]", "[\"mod_corte\",\"mod_coron\",\"dir_corte\"]"] {type:"raw"}



labels = ["NE","SE","SW","NW"] #@param ["[\"NE\",\"SE\",\"SW\",\"NW\"]", "[\"1\",\"2\"]", "[\"1\",\"2\",\"3\",\"4\",\"5\",\"6\"]", "[\"1\",\"2\",\"3\",\"4\",\"5\",\"6\",\"7\",\"8\",\"9\",\"10\"]"] {type:"raw"}





#filter only data above a threshold

mask=join[join[var_threshold]>=threshold].dropna()

#cut in bins==wind quadrants exemple

Y=pd.cut(x = mask[dir_obs], bins = len(labels), labels = labels)



#independent variables. Also observed variables!! if you wish

X=mask[dir_pred2]





#we do not scale!!

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,)



#select classification

clf2 = RandomForestClassifier().fit(x_train,y_train) 

y_pred=clf2.predict(x_test)



#plot results

print(classification_report(y_test,y_pred))

y_pred_df=pd.DataFrame({"dir_pred":y_pred},index=y_test.index)



#plot results

table=pd.crosstab(y_test,y_pred_df["dir_pred"], margins=True,)

table_columns2=pd.crosstab(y_test,y_pred_df["dir_pred"], margins=True,normalize="columns")

table_index=pd.crosstab(y_test,y_pred_df["dir_pred"], margins=True,normalize="index")





fig, axs = plt.subplots(3,figsize = (8,10))

sns.heatmap(table,annot=True,ax=axs[0],cmap="YlGnBu",fmt='.0f',)

sns.heatmap(table_columns2,annot=True,ax=axs[1],cmap="YlGnBu",fmt='.0%')

sns.heatmap(table_index,annot=True,ax=axs[2],cmap="YlGnBu",fmt=".0%")

plt.show()

#@title Plot results

delete_ten_minutes = False #@param {type:"boolean"}

show_graph = False #@param {type:"boolean"}

date_input = '2020-06-10' #@param {type:"date"}

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



model[dir_pred0+"_l"]=pd.cut(pd.concat([join[dir_obs],model[dir_pred0]]),bins=len(labels),labels=labels)[-24:].astype(str)

model[dir_obs+"_decisiontree"]=clf1.predict(model[dir_pred1])

model[dir_obs+"_randomforest"]=clf2.predict(model[dir_pred2])







#station results

variables_station=["spd_o_corte","std_spd_o_corte","gust_spd_o_corte","dir_o_corte"]

param=["param=81","param=10009","param=10003","param=82"]



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





if delete_ten_minutes:

  final_s=final[["spd_o_corte","gust_spd_o_corte","dir_o_corte",dir_pred0,dir_pred0+"_l",

                 dir_obs+"_decisiontree",dir_obs+"_randomforest"]].dropna()

else:

  final_s=final[["spd_o_corte","gust_spd_o_corte","dir_o_corte",dir_pred0,dir_pred0+"_l",

                 dir_obs+"_decisiontree",dir_obs+"_randomforest"]]

final_s
#@title Plot results

q_df=final[[dir_obs+"_decisiontree",dir_obs+"_randomforest"]].dropna()

pd.set_option('max_colwidth', 2000)

quantum_randomforest=[]

quantum_decisiontree=[]

formatter="{'.0%'}".format

for i in range(0, len(q_df.index)):

  quantum_randomforest.append(table_columns2[q_df[dir_obs+"_randomforest"][i]].map("{:.0%}".format))

  quantum_decisiontree.append(table_columns1[q_df[dir_obs+"_decisiontree"][i]].map("{:.0%}".format))

  

quantum_fi=pd.DataFrame({dir_obs+"_decision tree":quantum_decisiontree,dir_obs+"_randomforest":quantum_randomforest}, index=q_df.index)

quantum_fi
#@title Select time forecast

hour = 17 #@param {type:"slider", min:0, max:23, step:1}

H_resolution = True #@param {type:"boolean"}

variable_met = "dir" 



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







df_n[variable_met]= df_n[variable_met].astype(int)





df_n[variable_met]=df_n[variable_met].astype(str)

kml = simplekml.Kml()

df_n.apply(lambda X: kml.newpoint(name=X[variable_met], coords=[( X["lon"],X["lat"])]) ,axis=1)



#add Cortegada wind variables if variable_met mod or wind_gust

description="units degrees\n"+quantum_fi.columns[0]+" "+str(quantum_fi.iloc[hour,0])[:-15]+"\n"+quantum_fi.columns[1]+" "+str(quantum_fi.iloc[hour,1])[:-15]



#add Cortegada dir

string=final.index.strftime("%Y-%m-%d")[0]+" "+str(hour)+":00:00"

if dir_obs[-5:]=="coron":

  description="*"

kml.newpoint(name=str(round(final['dir_o_corte'].loc[string],0)), description=description,coords=[(-8.7836,42.6255)]) 

 

 

#add Coron

description="units degrees\n"+quantum_fi.columns[0]+" "+str(quantum_fi.iloc[hour,0])[:-15]+"\n"+quantum_fi.columns[1]+" "+str(quantum_fi.iloc[hour,1])[:-15]

if dir_obs[-5:]=="corte":

  description="*"

kml.newpoint(name="Coron",description=description,coords=[(-8.8046,42.5801)])   



#save results  

kml.save(today+"H"+str(hour)+r+variable_met+".kml")