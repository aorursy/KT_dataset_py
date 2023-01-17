#load trained MNN,lightgbm and randomforest AI models. MNN load with Keras. lighgbm and randomforest from pickle files

from tensorflow.keras.models import  load_model
import pickle
model_neural = load_model('/kaggle/input/mnn-files/model.h5')
model_lightgbm=pickle.load(open("/kaggle/input/visibility-threshold-5000m-lightgbm/vis_5000_lightgbm", 'rb')) 
model_randomforest=pickle.load(open("/kaggle/input/visibility-5000-randomforest/vis_5000_Randomforest", 'rb')) 

#Load database dependent variables and scale it. Always same scale than AI model: MinMaxScaler !!
#input variables are different for each model
import pandas as pd
master=pd.read_csv("../input/meteorological-model-versus-real-data/vigo_model_vs_real.csv",index_col="datetime",parse_dates=True)

x_data_MNN=master[['dir_4K', 'lhflx_4K', 'mod_4K', 'prec_4K', 'rh_4K', 'visibility_4K',
        'mslp_4K', 'temp_4K', 'cape_4K', 'cfl_4K', 'cfm_4K', 'cin_4K',"wind_gust_4K",
       'conv_prec_4K']] 

x_data_lightgbm=master[["dir_4K", "lhflx_4K", "mod_4K", "prec_4K", "rh_4K", "visibility_4K",
 "mslp_4K", "temp_4K", "cape_4K", "cfl_4K", "cfm_4K", "cin_4K","conv_prec_4K"]]

#we need to scale input data from MNN and lightgbm algorithm. 
from sklearn.preprocessing import MinMaxScaler
scaler_MNN =MinMaxScaler().fit(x_data_MNN)  
scaler_lightgbm =MinMaxScaler().fit(x_data_lightgbm)  
from datetime import datetime, timedelta, date
from urllib.request import urlretrieve
import xarray as xr

#creating the string_url


#analysis day= Yesterday. Time 00:00Z. 
datetime_str = (date.today()-timedelta(days = 1)).strftime('%Y%m%d')

#day to forecast 1= D+1 , 2 =D+2 and so on 
forecast=1

date_anal = datetime.strptime(datetime_str,'%Y%m%d')
date_fore=(date_anal+timedelta(days=forecast)).strftime('%Y-%m-%d')

#variables string type to perform url. The same variables as model (AI)

head="http://mandeo.meteogalicia.es/thredds/ncss/wrf_2d_04km/fmrc/files/"
text1="/wrf_arw_det_history_d03_"+datetime_str+"_0000.nc4?"
met_var="var=dir&var=lhflx&var=mod&var=prec&var=rh&var=visibility&var=mslp&var=temp&var=cape&var=cfl&var=cfm&var=cin&var=wind_gust&var=conv_prec&"
coordenates="latitude=42.2&longitude=-8.63&"
scope1="time_start="+date_fore+"T00%3A00%3A00Z&"
scope2="time_end="+date_fore+"T23%3A00%3A00Z&accept=netcdf"

#add all the string variables
url=head+datetime_str+text1+met_var+coordenates+scope1+scope2

#load the actual model from Meteogalicia database and transform as pandas dataframe
urlretrieve(url,"model")
df=xr.open_dataset("model").to_dataframe().set_index("time").loc[:, 'dir':]

#select Input variables for each AI model
x_MNN=df[['dir', 'lhflx', 'mod', 'prec', 'rh', 'visibility', 'mslp', 'temp',
       'cape', 'cfl', 'cfm', 'cin', 'wind_gust', 'conv_prec']]
x_lightgbm=df[['dir', 'lhflx', 'mod', 'prec', 'rh', 'visibility', 'mslp', 'temp',
       'cape', 'cfl', 'cfm', 'cin', 'conv_prec']]
x_randomforest=df[['dir', 'lhflx', 'mod', 'prec', 'rh', 'visibility', 'mslp', 'temp',
       'cape', 'cfl', 'cfm', 'cin', 'wind_gust', 'conv_prec']]

#scaler fitted before. Randomforest do not need to scale 
x_model_MNN=scaler_MNN.transform(x_MNN)
x_model_lightgbm=scaler_lightgbm.transform(x_lightgbm)

#define threshold visibility: the same as model trained
threshold1=5000
threshold2=5000
threshold3=5000

#define model threshold normalized same as trained AI model
threshold_nor1=0.5
threshold_nor2=0.5

#define column with model(AI) results from meteorological model data depèndent variables
df["visibility_neural"]=["low_"+ str(threshold1) if c < threshold_nor1 else "upper_" +str(threshold1) for c in model_neural.predict(x_model_MNN)]
df["visibility_lightgbm"]=["low_"+ str(threshold2) if c < threshold_nor2 else "upper_" +str(threshold2) for c in model_lightgbm.predict(x_model_lightgbm)]
df["visibility_randomforest"]=["low_"+ str(threshold3) if c==1 else "upper_" +str(threshold3) for c in model_randomforest.predict(x_randomforest)]

#select columns from df and change units pressure, temperature and normalize visibility model variable

df1=df.loc[:,("dir","mod","wind_gust","visibility","visibility_neural","visibility_lightgbm","visibility_randomforest","prec","temp","rh","mslp")]
df1.loc[:,("mslp")]=round(df1.loc[:,("mslp")]/100)
df1.loc[:,("temp")]=round(df1.loc[:,("temp")]-273.15)
df1.loc[:,("visibility")]=[9999 if c >9999 else round(c) for c in df1.visibility]
#Setting up the url 
head="https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

#Select station OACI code
station="?station=LEVX"

#Select day=today

today_year=datetime.today().strftime('%Y')
today_month=datetime.today().strftime('%m')
today_day=datetime.today().strftime('%d')
date="&year1="+today_year+"&month1="+today_month+"&day1="+today_day+"&year2="+today_year+"&month2="+today_month+"&day2="+today_day+"&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2"

#Select variables from Metar report
var="&data=drct"
variables=["drct","sknt","gust","vsby","wxcodes","tempf","relh","alti"]

url=head+station+var+date
df_metar=pd.read_csv(urlretrieve(url)[0])
for variable in variables:
    var="&data="+variable
    url=head+station+var+date
    df=pd.read_csv(urlretrieve(url)[0])
    df_metar[variable]=df[df.columns[-1]]
try:
    df_metar["dir_o"]=df_metar["drct"]
except:
    df_metar["dir_o"]=-9999
try:
    df_metar["mod_o"]=round(df_metar["sknt"]*0.51444)
except:
    df_metar["mod_o"]="M"
try:
    df_metar["wind_gust_o"]=["M" if c=="M" else round(c*0.51444) for c in df_metar.gust]
except:
    df_metar["wind_gust_o"]=-9999
try:
    df_metar['visibility_o']=round(df_metar["vsby"]*1609.344)
except:
    df_metar['visibility_o']=df_metar["vsby"]
try:
    df_metar["temp_o"]=(df_metar["tempf"]-32.0)*(5/9)
except:
    df_metar["temp_o"]=df_metar["tempf"]
try:
    df_metar["rh_o"]=round(df_metar["relh"])
except:
    df_metar["rh_o"]=df_metar["relh"]
try:
    df_metar["mslp_o"]=round(df_metar["alti"]*33.86)
except:
    df_metar["mslp_o"]=df_metar["alti"]
    

df_metar=df_metar.drop(["drct","sknt","gust","vsby","tempf","relh","alti","station"],axis=1)
df_metar=df_metar.rename(columns={"valid":"time"})
df_metar["time"]=pd.to_datetime(df_metar["time"])
df_metar=df_metar.set_index("time")
df_global=pd.concat([df_metar,df1[["dir","mod","wind_gust","visibility","visibility_neural","visibility_lightgbm","visibility_randomforest","prec","temp","rh","mslp"]]],axis=1)
#fig, axs = plt.subplots(3,figsize = (15,15))
g1=df_global[["mslp","mslp_o"]].plot().grid(True, which='both')
g2=df_global[["mslp","mslp_o"]].dropna().plot().grid(True, which='both')
g3=df_global[["temp","temp_o"]].plot().grid(True, which='both')
g4=df_global[["temp","temp_o"]].dropna().plot().grid(True, which='both')
g5=df_global[["visibility","visibility_o"]].plot().grid(True, which='both')
g6=df_global[["visibility","visibility_o"]].dropna().plot().grid(True, which='both')
g7=df_global[["mod","mod_o"]].plot().grid(True, which='both')
g8=df_global[["mod","mod_o"]].dropna().plot().grid(True, which='both')
g9=df_global[["dir","dir_o"]].plot().grid(True, which='both')
g10=df_global[["dir","dir_o"]].dropna().plot().grid(True, which='both')


df_global[["visibility_o","visibility","visibility_neural","visibility_lightgbm","visibility_randomforest","wxcodes","prec","rh","rh_o"]]
import seaborn as sns
g=sns.pairplot(df_global[["visibility","visibility_o","visibility_neural"]].dropna(), hue = "visibility_neural", diag_kind = 'hist', height = 4)
