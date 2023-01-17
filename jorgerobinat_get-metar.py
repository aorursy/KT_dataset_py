from datetime import datetime, timedelta, date

import pandas as pd

from urllib.request import urlretrieve

#Setting up the url 

head="https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"



#Select station ICAO code

station="?station=TJSJ"



#Select  first date



year1="2005"

month1="01"

day1="30"



#Last   date

year2="2019"

month2="10"

day2="30"





date="&year1="+year1+"&month1="+month1+"&day1="+day1+"&year2="+year2+"&month2="+month2+"&day2="+day2+"&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2"



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

    df_metar["temp_o"]=(df_metar["tempf"]-32.0)*(5/9).astype(int)

except:

    df_metar["temp_o"]=((pd.to_numeric(df_metar["tempf"][df_metar["tempf"]!="M"])-32)*(5/9)).astype(int)

try:

    df_metar["rh_o"]=round(df_metar["relh"])

except:

    df_metar["rh_o"]=df_metar["relh"]

try:

    df_metar["mslp_o"]=round(df_metar["alti"]*33.86)

except:

    df_metar["mslp_o"]=df_metar["alti"]

    



df_metar=df_metar.drop(["drct","sknt","gust","vsby","relh","alti","station"],axis=1)

df_metar=df_metar.rename(columns={"valid":"time"})

df_metar["time"]=pd.to_datetime(df_metar["time"])

df_metar=df_metar.set_index("time")
df_metar

df_metar["temp_o"].to_csv("TJSJ.csv")