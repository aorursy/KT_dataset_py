#Imports

import numpy as np 
import pandas as pd 
import seaborn as sns
import geopandas as gpd
import os
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import datetime
import matplotlib.dates as mdates

#Filter warnings
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('../input/covid19spain/datos_ccaas.csv')
df.head()
#Read Datasets
df = pd.read_csv('../input/covid19-spain-report-27032020/serie_historica_acumulados.csv',error_bad_lines=False)  #Spain regions data
grupos_edad = pd.read_csv("../input/covid19-spain-report-27032020/grupos_edad.csv") #Spain regions ages of population
sf = gpd.read_file('../input/shapees/ESP_adm1.shp') #Regions Geometry Map 

df = df.rename({'FECHA': 'Fecha', 'CASOS': 'Casos'}, axis=1)  # new method


population = {
                 'Andalucía':8414240,
                 'Aragón':1319291,
                 'Principado de Asturias':1022800,
                 'Islas Baleares':1149460,
                 'Islas Canarias':2153389,
                 'Cantabria':581078,
                 'Castilla-La Mancha':2032863,
                 'Castilla y León':2399548,
                 'Cataluña':7675217,
                 'Ceuta y Melilla':171264,
                 'Comunidad Valenciana':5003769,
                 'Extremadura':1067710,
                 'Galicia':2699499,
                 'Comunidad de Madrid':6663394,
                 'Región de Murcia':1493898,
                 'Comunidad Foral de Navarra':654214,
                 'País Vasco':2207776,
                 'La Rioja':316798    
} # https://www.ine.es/jaxiT3/Tabla.htm?t=2853&L=0

population = pd.DataFrame.from_dict(population,orient='index',columns=["Población"]) #Regions population

df["Fecha"] = pd.to_datetime(df["Fecha"],format="%d/%m/%Y") #--
#df["Fecha"] = df["Fecha"].dt.date #-- Transform to date format

df["Casos"] = pd.Series(np.where(~df["Casos"].isnull(),df["Casos"],df["PCR+"] + df["TestAc+"]))
last_data = df.tail(19).copy() #Data from the last day
last_data = last_data.replace({'CCAA' : {'ML':'CE'}}).groupby('CCAA', sort=False).sum() #Join Ceuta and Melilla data, beacuse its only one in the map
#Styles
sns.set(style="darkgrid",palette="pastel", color_codes=True)
sns.mpl.rc("figure", figsize=(17,10))
def set_date_format(ax,day=2,month=1):
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=day)) 
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b')) 
def anotate_values(ax,df,var_name,f_type=lambda x: x,decimals = 2):
    """
    Annotate value on the center of the map
    """
    try:
        df.apply(lambda x: ax.annotate(s=f_type(round(x[var_name], decimals)), xy=x.geometry.centroid.coords[0], ha='center',color="Black"),axis=1)
        return ax
    except: 
        pass

def create_evolution(data,ax):
    ax.stackplot(data["Fecha"],data["Casos"],color="skyblue",labels=["Casos"]) 
    ax.stackplot(data["Fecha"], data["Hospitalizados"],color="lightcoral",labels=["Hospitalizados"])
    ax.stackplot(data["Fecha"], data["Recuperados"],color="lightgreen",labels=["Recuperados"],alpha=.5)
    ax.stackplot(data["Fecha"], data["Fallecidos"],color="Black",labels=["Fallecidos"],alpha=.6)

    ax.text(x=data["Fecha"].max(),y=data["Casos"].max(),s="{}".format(int(data["Casos"].max())),fontsize=12,ha="center")
    ax.text(x=data["Fecha"].max(),y=data["Recuperados"].max(),s="{}".format(int(data["Recuperados"].max())),fontsize=12,ha="center")
    ax.text(x=data["Fecha"].max(),y=data["Hospitalizados"].max(),s="{}".format(int(data["Hospitalizados"].max())),fontsize=12,ha="center")
    ax.text(x=data["Fecha"].max(),y=data["Fallecidos"].max(),s="{}".format(int(data["Fallecidos"].max())),fontsize=12,ha="center")

    ax.set_xlim([data["Fecha"].min(),data["Fecha"].max()])
    ax.legend(loc='upper left')
    set_date_format(ax)

    #ax.set_xticks(rotation=45,ha="right")
    return ax
def plot_col_map(df,col_name,f_type=lambda x: x,title="",color="Reds",decimals = 2):
    """
    Plot map for the column col_name from dataset df
    """
    ax = df.plot(column=col_name, cmap=color,legend=True,edgecolor='Grey')
    ax = anotate_values(ax,result,col_name,f_type,decimals)
    ax.set_title(title)
    ax.set_axis_off()
    ax.plot()
def plot_col_map2(df,col_name,f_type=lambda x: x,title="",color="Reds",decimals = 2,date=None):
    """
    Plot map for the column col_name from dataset df
    """
    ax = df.plot(column=col_name, cmap=color,legend=True,edgecolor='Grey')
    ax = anotate_values(ax,df,col_name,f_type,decimals)
    ax.set_title(title)
    ax.set_axis_off()
    plt.suptitle("{}".format(date),fontsize=18)
    return ax

#Change the name of Regions
last_data.rename({'AN':'Andalucía',
                 'AR':'Aragón',
                 'AS':'Principado de Asturias',
                 'IB':'Islas Baleares',
                 'CN':'Islas Canarias',
                 'CB':'Cantabria',
                 'CM':'Castilla-La Mancha',
                 'CL':'Castilla y León',
                 'CT':'Cataluña',
                 'CE':'Ceuta y Melilla',
                 'VC':'Comunidad Valenciana',
                 'EX':'Extremadura',
                 'GA':'Galicia',
                 'MD':'Comunidad de Madrid',
                 'MC':'Región de Murcia',
                 'NC':'Comunidad Foral de Navarra',
                 'PV':'País Vasco',
                 'RI':'La Rioja'
                },inplace=True)
sf.set_index("NAME_1",drop=True,inplace=True)

result = pd.concat([last_data, sf["geometry"],population], axis=1, sort=False) #Join Data from last date, geometry and population
result = gpd.GeoDataFrame(result)

result.reset_index(inplace=True)

#Fix Canary Islands position to display them closer to Iberian Peninsula
import shapely
from shapely.geometry import MultiPolygon
canarias_geom = result.loc[result["index"].str.contains("Islas Canarias"),"geometry"].values[0]

new_pol = []
for pol in canarias_geom:
    new_pol.append(shapely.affinity.translate(pol, xoff=7, yoff=5))
result.loc[result["index"].str.contains("Islas Canarias"),"geometry"] = gpd.GeoDataFrame(geometry=[MultiPolygon(new_pol)]).geometry.values
print("Datos actualizados a / Data updated to: {}".format(df["Fecha"].max()))

spain = df.groupby("Fecha").sum()
spain.reset_index(inplace=True)

"""
Plot spain evolution

"""

fig, ax = plt.subplots()

create_evolution(spain,ax)
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(4, 2)
fig = plt.figure(figsize=(14,20))

fig.suptitle("España", fontsize=20)


spain["new_casos"] = spain["Casos"].diff()
spain["new_hosp"] = spain["Hospitalizados"].diff()
spain["new_uci"] = spain["UCI"].diff()
spain["new_fal"] = spain["Fallecidos"].diff()
spain["new_rec"] = spain["Recuperados"].diff()


#____Plot Confirmed cases of the region___#

ax4 = fig.add_subplot(gs[0, 0])
plt.bar(x="Fecha",height="Casos",data=spain,color="skyblue")
ax4.yaxis.grid(True)
ax4.set_ylabel("Casos confirmados")
set_date_format(ax4,4)

ax41 = fig.add_subplot(gs[0, 1])
plt.bar(x="Fecha",height="new_casos",data=spain,color="skyblue")
ax41.yaxis.grid(True)
ax41.set_ylabel("Nuevos Casos")
set_date_format(ax41,4)

#____Plot hospitalized of the region___#

ax5 = fig.add_subplot(gs[1, 0])
plt.bar(x="Fecha",height="Hospitalizados",data=spain,color="lightcoral")
ax5.yaxis.grid(True)
set_date_format(ax5,4)
ax5.set_ylabel("Hospitalizados")


ax51 = fig.add_subplot(gs[1, 1])
plt.bar(x="Fecha",height="new_hosp",data=spain,color="lightcoral")
ax51.yaxis.grid(True)
set_date_format(ax51,4)
ax51.set_ylabel("Variación Hospitalizados")

#____Plot Revovered of the region___#

ax6 = fig.add_subplot(gs[2, 0])
plt.bar(x="Fecha",height="Recuperados",data=spain,color="lightgreen")
set_date_format(ax6,4)
ax6.yaxis.grid(True)
ax6.set_ylabel("Recuperados")

ax61 = fig.add_subplot(gs[2, 1])
plt.bar(x="Fecha",height="new_rec",data=spain,color="lightgreen")
set_date_format(ax61,4)
ax61.yaxis.grid(True)
ax61.set_ylabel("Nuevos Recuperados")

#____Plot Deaths of the region___#

ax7 = fig.add_subplot(gs[3, 0])
plt.bar(x="Fecha",height="Fallecidos",data=spain,color="Black")
set_date_format(ax7,4)
ax7.yaxis.grid(True)
ax7.set_ylabel("Fallecidos")

ax71 = fig.add_subplot(gs[3, 1])
plt.bar(x="Fecha",height="new_fal",data=spain,color="Black")
set_date_format(ax71,4)
ax71.yaxis.grid(True)
ax71.set_ylabel("Variación Fallecidos")



plt.subplots_adjust(hspace=.3)
df = df.replace({'CCAA' : {'ML':'CE'}}).groupby(['CCAA',"Fecha"], sort=False).sum()
df.reset_index(inplace=True)
df.replace({'CCAA':{'AN':'Andalucía',
                 'AR':'Aragón',
                 'AS':'Principado de Asturias',
                 'IB':'Islas Baleares',
                 'CN':'Islas Canarias',
                 'CB':'Cantabria',
                 'CM':'Castilla-La Mancha',
                 'CL':'Castilla y León',
                 'CT':'Cataluña',
                 'CE':'Ceuta y Melilla',
                 'VC':'Comunidad Valenciana',
                 'EX':'Extremadura',
                 'GA':'Galicia',
                 'MD':'Comunidad de Madrid',
                 'MC':'Región de Murcia',
                 'NC':'Comunidad Foral de Navarra',
                 'PV':'País Vasco',
                 'RI':'La Rioja'
                }},inplace=True)
df.set_index("CCAA",inplace=True,drop=True)
plot_col_map(result,"Casos",lambda x: int(x),"Número de casos por CCAA","Reds",0)
result['Casos_Población'] = result['Casos']/result['Población']
plot_col_map(result,"Recuperados",title="Número de recuperados",f_type=lambda x: int(x),color="YlGn",decimals=0)
result['Recuperados_Casos'] = (result['Recuperados']/result['Casos'])*100
plot_col_map(result,"Recuperados_Casos",title="Porcentaje de recuperados en cada CCAA",f_type=lambda x: "{} %".format(int(x)),color="YlGn",decimals=0)
plot_col_map(result,"Casos_Población",title="Proporción de casos en relación con la población de cada región",f_type=lambda x: x,color="Reds",decimals=5)
result['Fallecidos_Casos'] = result['Fallecidos'] / result['Casos']
plot_col_map(result,"Fallecidos_Casos",title="Proporción de casos en relación con la población de cada región",f_type=lambda x: x,color="Reds",decimals=2)
df_groups = df.copy()
df_groups.replace({'CCAA' : {'ML':'CE'}},inplace=True)
df_groups = df_groups.groupby(["CCAA","Fecha"]).sum()
df_groups.reset_index(inplace=True)

df_groups.replace({'CCAA' : {'AN':'Andalucía',
                 'AR':'Aragón',
                 'AS':'Principado de Asturias',
                 'IB':'Islas Baleares',
                 'CN':'Islas Canarias',
                 'CB':'Cantabria',
                 'CM':'Castilla-La Mancha',
                 'CL':'Castilla y León',
                 'CT':'Cataluña',
                 'CE':'Ceuta y Melilla',
                 'VC':'Comunidad Valenciana',
                 'EX':'Extremadura',
                 'GA':'Galicia',
                 'MD':'Comunidad de Madrid',
                 'MC':'Región de Murcia',
                 'NC':'Comunidad Foral de Navarra',
                 'PV':'País Vasco',
                 'RI':'La Rioja'
                }},inplace=True)
sf.reset_index(inplace=True)
grupos_edad = pd.read_csv("../input/covid19-spain-report-27032020/grupos_edad.csv") #Ages groups 
grupos_edad["CCAA"].unique()
grupos_edad.replace({'CCAA' : {
                 'MELILLA':'CEUTA',
                }},inplace=True)

grupos_edad.replace({'CCAA' : {
                 'ANDALUCIA':'Andalucía',
                 'ARAGON':'Aragón',
                 'ASTURIAS ':'Principado de Asturias',
                 'BALEARS ILLES':'Islas Baleares',
                 'CANARIAS':'Islas Canarias',
                 'CANTABRIA':'Cantabria',
                 'CASTILLA  LA MANCHA':'Castilla-La Mancha',
                 'CASTILLA Y LEON':'Castilla y León',
                 'CATALUNYA':'Cataluña',
                 'CEUTA':'Ceuta y Melilla',
                 'COMUNITAT VALENCIANA':'Comunidad Valenciana',
                 'EXTREMADURA':'Extremadura',
                 'GALICIA':'Galicia',
                 'MADRID COMUNIDAD DE':'Comunidad de Madrid',
                 'MURCIA REGION DE':'Región de Murcia',
                 'NAVARRA COMUNIDAD FORAL DE':'Comunidad Foral de Navarra',
                 'PAIS VASCO':'País Vasco',
                 'RIOJA LA':'La Rioja'
                }},inplace=True)

grupos_edad = grupos_edad.groupby(["CCAA","Edad_min","Edad_max"]).sum()
grupos_edad.reset_index(inplace=True)

df_groups["Fecha"] = pd.to_datetime(df_groups["Fecha"])
import matplotlib.gridspec as gridspec
sns.set_color_codes("pastel")


for ca in df_groups["CCAA"].unique():    
    data = df_groups.loc[df_groups["CCAA"] == ca]
    data["new_casos"] = data["Casos"].diff()
    data["new_hosp"] = data["Hospitalizados"].diff()
    data["new_uci"] = data["UCI"].diff()
    data["new_fal"] = data["Fallecidos"].diff()
    data["new_rec"] = data["Recuperados"].diff()
    
    gs = gridspec.GridSpec(7, 2)
    fig = plt.figure(figsize=(14,20))
    
    fig.suptitle(ca, fontsize=20)
    
    #____Plot MAP of the region___#
    
    ax1 = fig.add_subplot(gs[0, :])
    sf.loc[sf["NAME_1"] == ca,"geometry"].plot(ax=ax1)
    ax1.annotate(s=int(population.at[ca,"Población"]), xy=sf.loc[sf["NAME_1"] == ca].iloc[0].geometry.centroid.coords[0], ha='center',color="Black")
    ax1.set_axis_off()
    ax1.set_xlabel("Población")

    #____Plot Age groups of the region___#

    
    ax2 = fig.add_subplot(gs[1, :])
    sns.set_color_codes("pastel")
    sns.barplot(y="Total",x="Edad_min",data=grupos_edad.loc[grupos_edad["CCAA"] == ca],color="b")
    ax2.yaxis.grid(True)
    ax2.set_xlabel("Edad")
    ax2.set_ylabel("Población")
    
    #____Plot Evolution of the region___#
    
    ax3 = fig.add_subplot(gs[2, :])
    ax3 = create_evolution(data=data,ax=ax3)
    ax3.yaxis.grid(True)
    set_date_format(ax3,4)
     
   

    #____Plot Confirmed cases of the region___#

    ax4 = fig.add_subplot(gs[3, 0],sharex=ax3)
    plt.bar(x="Fecha",height="Casos",data=data,color="skyblue")
    ax4.yaxis.grid(True)
    ax4.set_ylabel("Casos confirmados")
    set_date_format(ax4,4)
    
    ax41 = fig.add_subplot(gs[3, 1],sharex=ax3)
    plt.bar(x="Fecha",height="new_casos",data=data,color="skyblue")
    ax41.yaxis.grid(True)
    ax41.set_ylabel("Nuevos Casos")
    set_date_format(ax41,4)

    #____Plot hospitalized of the region___#

    ax5 = fig.add_subplot(gs[4, 0],sharex=ax3)
    plt.bar(x="Fecha",height="Hospitalizados",data=data,color="lightcoral")
    ax5.yaxis.grid(True)
    set_date_format(ax5,4)
    ax5.set_ylabel("Hospitalizados")
    
    
    ax51 = fig.add_subplot(gs[4, 1],sharex=ax3)
    plt.bar(x="Fecha",height="new_hosp",data=data,color="lightcoral")
    ax51.yaxis.grid(True)
    set_date_format(ax51,4)
    ax51.set_ylabel("Variación Hospitalizados")
    
    #____Plot Revovered of the region___#

    ax6 = fig.add_subplot(gs[5, 0],sharex=ax3)
    plt.bar(x="Fecha",height="Recuperados",data=data,color="lightgreen")
    set_date_format(ax6,4)
    ax6.yaxis.grid(True)
    ax6.set_ylabel("Recuperados")
    
    ax61 = fig.add_subplot(gs[5, 1],sharex=ax3)
    plt.bar(x="Fecha",height="new_rec",data=data,color="lightgreen")
    set_date_format(ax61,4)
    ax61.yaxis.grid(True)
    ax61.set_ylabel("Nuevos Recuperados")

    #____Plot Deaths of the region___#

    ax7 = fig.add_subplot(gs[6, 0],sharex=ax3)
    plt.bar(x="Fecha",height="Fallecidos",data=data,color="Black")
    set_date_format(ax7,4)
    ax7.yaxis.grid(True)
    ax7.set_ylabel("Fallecidos")
    
    ax71 = fig.add_subplot(gs[6, 1],sharex=ax3)
    plt.bar(x="Fecha",height="new_fal",data=data,color="Black")
    set_date_format(ax71,4)
    ax71.yaxis.grid(True)
    ax71.set_ylabel("Variación Fallecidos")
    


    plt.subplots_adjust(hspace=.3)
    


    plt.plot()

spain = df.groupby("Fecha").sum().copy()
spain.reset_index(inplace=True)


china = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
china = china.loc[china["Country/Region"].str.contains("China")]
china["Date"] = pd.to_datetime(china["ObservationDate"])
china["Date"] = pd.to_datetime(china["Date"].dt.date)
china = china.groupby("Date").sum()
china.reset_index(inplace=True)


italy = pd.read_csv("../input/covid19-in-italy/covid19_italy_region.csv")
italy["Date"] = pd.to_datetime(italy["Date"],format="%Y-%m-%d")
italy["Date"] = pd.to_datetime(italy["Date"].dt.date)
italy = italy.groupby("Date").sum()
italy.reset_index(inplace=True)

countries = pd.DataFrame([])
countries["spain_date"] = pd.to_datetime(spain["Fecha"])
countries["spain_positive"] = spain["Casos"]
countries["spain_recovered"] = spain["Recuperados"]
countries["spain_deaths"] = spain["Fallecidos"]


countries = pd.concat([countries, china[["Date","Confirmed","Recovered","Deaths"]]], axis=1)
countries = pd.concat([countries, italy[["Date","TotalPositiveCases","Recovered","Deaths"]]], axis=1)



countries.columns = ["spain_date","spain_positive","spain_recovered","spain_deaths",
                     "china_date","china_positive","china_recovered","china_deaths",
                     "italy_date","italy_positive","italy_recovered","italy_deaths"
                    ]


countries["spain_active"] = countries["spain_positive"] - (countries["spain_recovered"] + countries["spain_deaths"])
countries["china_active"] = countries["china_positive"] - (countries["china_recovered"] + countries["china_deaths"])
countries["italy_active"] = countries["italy_positive"] - (countries["italy_recovered"] + countries["italy_deaths"])
countries.reset_index(inplace=True)
countries.head()
plt.figure(figsize=(35,10))

ax = sns.lineplot(data=countries,x="spain_date",y="spain_positive",color="Blue")
ax = sns.lineplot(ax=ax,data=countries,x="italy_date",y="italy_positive",color="Red")
ax = sns.lineplot(ax=ax,data=countries,x="china_date",y="china_positive",color="Orange")

ax.scatter(x=countries["italy_date"],y=countries["italy_positive"],s=countries["italy_active"]/10,label="Italia Active cases",color="Red",alpha=0.3)
ax.scatter(x=countries["spain_date"],y=countries["spain_positive"],s=countries["spain_active"]/10,label="Spain Active cases",color="Blue",alpha=0.3)
ax.scatter(x=countries["china_date"],y=countries["china_positive"],s=countries["china_active"]/10,label="China Active cases",color="Orange",alpha=0.3)

ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1)) 
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d\n%a'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('\n\n%B'))

ax.grid(which='minor', alpha=0.5)
ax.grid(which='major', alpha=1)

ax.set_ylabel("Confirmed And Active Cases")
ax.set_xlabel("Days")
plt.legend(loc='upper left')
plt.plot()
plt.figure(figsize=(20,10))

ax = sns.lineplot(data=countries,x="index",y="spain_positive",color="Blue")
ax = sns.lineplot(ax=ax,data=countries,x="index",y="italy_positive",color="Red")
ax = sns.lineplot(ax=ax,data=countries,x="index",y="china_positive",color="Orange")

ax.scatter(x=countries["index"],y=countries["italy_positive"],s=countries["italy_active"]/10,label="Italia Active cases",color="Red",alpha=0.3)
ax.scatter(x=countries["index"],y=countries["spain_positive"],s=countries["spain_active"]/10,label="Spain Active cases",color="Blue",alpha=0.3)
ax.scatter(x=countries["index"],y=countries["china_positive"],s=countries["china_active"]/10,label="China Active cases",color="Orange",alpha=0.3)

ax.grid(which='minor', alpha=0.5)
ax.grid(which='major', alpha=1)


ax.set_ylabel("Active Cases")
ax.set_xlabel("Days")
plt.legend(loc='upper left')
plt.plot()
plt.figure(figsize=(35,10))


ax = sns.lineplot(data=countries,x="spain_date",y="spain_active",color="Blue",label="Spain")
ax = sns.lineplot(ax=ax,data=countries,x="italy_date",y="italy_active",color="Red",label="Italy")
ax = sns.lineplot(ax=ax,data=countries,x="china_date",y="china_active",color="Orange",label="China")


from datetime import date
china_lockdown = date(2020,1,23)
spain_lockdown = date(2020,3,14)
italy_lockdown = date(2020,3,9)

china_value = countries.loc[countries["china_date"] == china_lockdown]["china_active"]
spain_value = countries.loc[countries["spain_date"] == spain_lockdown]["spain_active"]
italy_value = countries.loc[countries["italy_date"] == italy_lockdown]["italy_active"]


ax.scatter(x=china_lockdown,y=china_value,color="Orange",label="China Lockdown",zorder=5,marker='X')
ax.text(x=china_lockdown,y=china_value,s="Lockdown {} -> {}".format(china_lockdown,int(china_value)),fontsize=12,ha="left")

ax.scatter(x=spain_lockdown,y=spain_value,color="Blue",label="Spain Lockdown",zorder=5,marker='X')
ax.text(x=spain_lockdown,y=spain_value,s="Lockdown {} -> {}".format(spain_lockdown,int(spain_value)),fontsize=12,ha="left",)

ax.scatter(x=italy_lockdown,y=italy_value,color="Red",label="Italy Lockdown",zorder=5, marker='X')
ax.text(x=italy_lockdown,y=italy_value,s="Lockdown {} -> {}".format(italy_lockdown,int(italy_value)),fontsize=12,ha="right")


ax.text(x=countries.iloc[countries["spain_active"].idxmax()]["spain_date"],y=countries["spain_active"].max(),s="{}".format(int(countries["spain_active"].max())),fontsize=12,ha="center")
ax.text(x=countries.iloc[countries["italy_active"].idxmax()]["italy_date"],y=countries["italy_active"].max(),s="{}".format(int(countries["italy_active"].max())),fontsize=12,ha="center")
ax.text(x=countries.iloc[countries["china_active"].idxmax()]["china_date"],y=countries["china_active"].max(),s="{}".format(int(countries["china_active"].max())),fontsize=12,ha="center")

ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1)) 
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d\n%a'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('\n\n%B'))

ax.grid(which='minor', alpha=0.5)
ax.grid(which='major', alpha=1)

ax.set_ylabel("Active Cases")
ax.set_xlabel("Date")
plt.legend(loc='upper left')

plt.plot()
countries['spain_new_positive'] = countries["spain_positive"].diff()
countries['italy_new_positive'] = countries["italy_positive"].diff()
countries['china_new_positive'] = countries["china_positive"].diff()
plt.figure(figsize=(45,15))

ax = sns.lineplot(data=countries,x="spain_date",y="spain_new_positive",color="Blue",label="Spain")
ax = sns.lineplot(ax=ax,data=countries,x="italy_date",y="italy_new_positive",color="Red",label="Italy")
ax = sns.lineplot(ax=ax,data=countries,x="china_date",y="china_new_positive",color="Orange",label="China")
#plt.legend(loc='upper left')
ax.set_ylabel("New Positives Cases")
ax.set_xlabel("Date")

china_value = countries.loc[countries["china_date"] == china_lockdown]["china_new_positive"]
spain_value = countries.loc[countries["spain_date"] == spain_lockdown]["spain_new_positive"]
italy_value = countries.loc[countries["italy_date"] == italy_lockdown]["italy_new_positive"]


ax.scatter(x=china_lockdown,y=china_value,color="Orange",label="China Lockdown",zorder=5,marker='X')
ax.text(x=china_lockdown,y=china_value,s="Lockdown {} -> {}".format(china_lockdown,int(china_value)),fontsize=12,ha="left")

ax.scatter(x=spain_lockdown,y=spain_value,color="Blue",label="Spain Lockdown",zorder=5,marker='X')
ax.text(x=spain_lockdown,y=spain_value,s="Lockdown {} -> {}".format(spain_lockdown,int(spain_value)),fontsize=12,ha="left")

ax.scatter(x=italy_lockdown,y=italy_value,color="Red",label="Italy Lockdown",zorder=5, marker='X')
ax.text(x=italy_lockdown,y=italy_value,s="Lockdown {} -> {}".format(italy_lockdown,int(italy_value)),fontsize=12,ha="right")

ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1)) 
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d\n%a'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('\n\n%B'))

ax.grid(which='minor', alpha=0.5)
ax.grid(which='major', alpha=1)

ax.legend()
ax.plot()
plt.figure(figsize=(30,10))

ax = sns.lineplot(data=countries,x="spain_date",y="spain_new_positive",color="Blue",label="Spain")

ax.set_ylabel("New Positives Cases")
ax.set_xlabel("Date")

spain_value = countries.loc[countries["spain_date"] == spain_lockdown]["spain_new_positive"]

ax.scatter(x=spain_lockdown,y=spain_value,color="Blue",label="Spain Lockdown",zorder=5,marker='X')
ax.text(x=spain_lockdown,y=spain_value,s="Lockdown {} -> {}".format(spain_lockdown,int(spain_value)),fontsize=12,ha="left")

ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1)) 
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d\n%a'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('\n\n%B'))

ax.grid(which='minor', alpha=0.5)
ax.grid(which='major', alpha=1)

ax.legend()
ax.plot()
global_data = pd.read_csv("../input/covid19-coronavirus/2019_nCoV_data.csv")

global_data["Date"] = pd.to_datetime(global_data["Date"])
global_data["Date"] = pd.to_datetime(global_data["Date"].dt.date)



global_data = global_data.replace({"Country":{
    "Mainland China": "China"
}})

global_data = global_data.groupby(["Country","Date",]).sum()
global_data.reset_index(inplace=True)

top_countries = global_data.groupby("Country").max().sort_values(by=['Confirmed'],ascending=False).reset_index()["Country"][:10]
top_countries = global_data.loc[global_data["Country"].isin(top_countries.to_list())]

top_countries = top_countries.sort_values(["Confirmed"], ascending = False)

sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

#ax = sns.lineplot(data=top_countries,x="Date",y="Confirmed",hue="Country")
ax = sns.FacetGrid(data=top_countries,row="Country",hue="Country",aspect=5,height=5)
ax.map(sns.lineplot,"Date","Confirmed")
ax.map(plt.fill_between,"Date","Confirmed")

ax.fig.subplots_adjust(hspace=.10)
ax.set(ylim=(0,top_countries["Confirmed"].max() ))
plt.xticks(rotation=45,ha='right')
plt.plot(figsize=(40,40))