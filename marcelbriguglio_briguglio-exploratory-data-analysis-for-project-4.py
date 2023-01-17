!pip install seaborn --upgrade # added to update seaborn to 0.11.0 (the latest version)
# Importation of nesscary packages

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as numpy

import geopandas 

import pysal

import plotly.express as px

print(sns.__version__)
data_download = pd.read_csv("../input/no2-files/train.csv",encoding="mac_latin2")
    #Region 1- Northeast

        #Division 1- New England

data_R1_D1 = data_download[(data_download["State"] == "CT") |  (data_download["State"] == "ME")|

                   (data_download["State"] == "MA") | (data_download["State"] == "NH") | 

                   (data_download["State"] == "RI") | (data_download["State"] == "VT")]



                   

        #Division 2- Mid-Atlantic 



data_R1_D2 = data_download[(data_download["State"] == "NJ") |  (data_download["State"] == "NY")| 

                     (data_download["State"] == "PA")]









data_R1_D1 = data_R1_D1.assign(Division = 1)

data_R1_D2 = data_R1_D2.assign(Division = 2)



data_R1 = pd.concat([data_R1_D1, data_R1_D2], axis = 0)





data_R1 = data_R1.assign(Region = 1)



    #Region 2- Midwest

        #Division 3- East North Central

data_R2_D3 = data_download[(data_download["State"] == "IL") |  (data_download["State"] == "IN")|

                   (data_download["State"] == "MI") | (data_download["State"] == "OH") | 

                    (data_download["State"] == "WI")]







        #Division 4- West North Central



data_R2_D4 =  data_download[(data_download["State"] == "IA") |  (data_download["State"] == "KS")|

                   (data_download["State"] == "MN") | (data_download["State"] == "MO") | 

                   (data_download["State"] == "NE") | (data_download["State"] == "ND") | 

                   (data_download["State"] == "SD")]





data_R2_D3 = data_R2_D3.assign(Division = 3)



data_R2_D4 = data_R2_D4.assign(Division = 4)



data_R2 = pd.concat([data_R2_D3, data_R2_D4], axis = 0)

data_R2 = data_R2.assign(Region = 2)





    #Region 3-South

        #Divison 5- South Atlantic

data_R3_D5 = data_download[(data_download["State"] == "DE") |  (data_download["State"] == "FL")|

                   (data_download["State"] == "GA") | (data_download["State"] == "MD") | 

                   (data_download["State"] == "NC") | (data_download["State"] == "SC") | 

                   (data_download["State"] == "VA") | (data_download["State"] == "DC") |

                   (data_download["State"] == "WV")]



        #Division 6- East South Central 



data_R3_D6 =  data_download[(data_download["State"] == "AL") |  (data_download["State"] == "KY")|

                   (data_download["State"] == "MS") | (data_download["State"] == "TN")]

        #Division 7- West North Central



data_R3_D7 =   data_download[(data_download["State"] == "AR") |  (data_download["State"] == "LA")|

                   (data_download["State"] == "OK") | (data_download["State"] == "TX")]



data_R3_D5 = data_R3_D5.assign(Division = 5)

data_R3_D6 = data_R3_D6.assign(Division = 6) 

data_R3_D7 = data_R3_D7.assign(Division = 7)



data_R3= pd.concat([data_R3_D5, data_R3_D6, data_R3_D7], axis = 0)

data_R3 = data_R3.assign(Region = 3)



    #Region 4

        # Division 8- Mountain

data_R4_D8 =  data_download[(data_download["State"] == "AZ") |  (data_download["State"] == "CO")|

                   (data_download["State"] == "ID") | (data_download["State"] == "MT") | 

                   (data_download["State"] == "NV") | (data_download["State"] == "NM") | 

                   (data_download["State"] == "UT") | (data_download["State"] == "WY")]





        #Division 9- Pacific



data_R4_D9 =  data_download[(data_download["State"] == "AK") |  (data_download["State"] == "CA")|

                   (data_download["State"] == "HI") | (data_download["State"] == "OR")|

                    (data_download["State"] == "WA")]



data_R4_D8 = data_R4_D8.assign(Division = 8)

data_R4_D9 = data_R4_D9.assign(Division = 9)





data_R4= pd.concat([data_R4_D8, data_R4_D9], axis = 0)

data_R4 = data_R4.assign(Region = 4)



#All data with region and Divisions



data_all = pd.concat([data_R1,data_R2,data_R3,data_R4], axis=0)
data_all_plot = data_all







data_all_plot["Region"] = data_all_plot["Region"].replace(to_replace= 1, value = "Northeast")

data_all_plot["Region"]= data_all_plot["Region"].replace(to_replace= 2, value = "Midwest")

data_all_plot["Region"]= data_all_plot["Region"].replace(to_replace= 3, value = "South")

data_all_plot["Region"]= data_all_plot["Region"].replace(to_replace= 4, value = "West")





data_all_plot["Division"] = data_all_plot["Division"].replace(to_replace= 1, value = "1")

data_all_plot["Division"]= data_all_plot["Division"].replace(to_replace= 2, value = "2")

data_all_plot["Division"]= data_all_plot["Division"].replace(to_replace= 3, value = "3")

data_all_plot["Division"]= data_all_plot["Division"].replace(to_replace= 4, value = "4")

data_all_plot["Division"] = data_all_plot["Division"].replace(to_replace= 5, value = "5")

data_all_plot["Division"]= data_all_plot["Division"].replace(to_replace= 6, value = "6")

data_all_plot["Division"]= data_all_plot["Division"].replace(to_replace= 7, value = "7")

data_all_plot["Division"]= data_all_plot["Division"].replace(to_replace= 8, value = "8")

data_all_plot["Division"]= data_all_plot["Division"].replace(to_replace= 9, value = "9")
sns.set_style('darkgrid')

sns.set(rc={'figure.figsize':(20,8.27)}) # allow to see all states in plot

State_bw = sns.boxplot(x= "State",y= "Observed_NO2_ppb", data= data_all_plot)

State_sp = sns.stripplot(x= "State",y= "Observed_NO2_ppb", data= data_all_plot, color= 'black', alpha = .6)

plt.ylabel("NO\u2082 Concentration (ppb)")

plt.show()

plt.savefig("NO2_concentrations_State.png")
sns.set_style("darkgrid")

sns.set(rc={'figure.figsize':(11.7,8.27)})

Division_bw = sns.boxplot(x= "Division",y= "Observed_NO2_ppb", data= data_all_plot)

Division_sp = sns.stripplot(x= "Division",y= "Observed_NO2_ppb", data= data_all_plot, color= 'black', alpha = .6)



plt.ylabel("NO\u2082 Concentration (ppb)")

plt.show()

plt.savefig("NO2_concentrations_Division.png")
sns.set_style("darkgrid")

sns.set(rc={'figure.figsize':(11.7,8.27)})

Region_bw = sns.boxplot(x= "Region",y= "Observed_NO2_ppb", data= data_all_plot)

Region_sp = sns.stripplot(x= "Region",y= "Observed_NO2_ppb", data= data_all_plot, color= 'black', alpha = .6)

plt.ylabel("NO\u2082 Concentration (ppb)")

plt.show()

plt.savefig("NO2_concentrations_Region.png")
sns.set(rc={'figure.figsize':(20,8.27)}) # allow to see all states in plot





Monitors_Division_bw = sns.countplot(x= "State" , data= data_all_plot)

plt.ylabel("Number of Monitors")

plt.show()

plt.savefig("Monitor_locations_State.png")
Monitor_state=   data_all_plot['State'].value_counts()

state_list = data_all["State"].unique()

Monitor_plot = sns.distplot(Monitor_state, kde= False)





plt.xlim(0)

plt.ylim(0)

plt.xlabel("Number of Monitors") 

plt.ylabel("State Count")

plt.show()

plt.savefig("Monitor_Distribution_State.png")
sns.set(rc={'figure.figsize':(11.7,8.27)})



Monitors_Division_bw = sns.countplot(x= "Region" , data= data_all_plot)

plt.ylabel("Number of Monitors")

plt.show()

plt.savefig("Monitor_Locations_Region.png")
# Download map of 48 contigous states form US Census Bureau

# Download at https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html



states_basemap = geopandas.read_file("../input/us-map/cb_2018_us_state_500k.shp")
# Entire US

sns.set(style = "white", rc={'figure.figsize':(30,20)}) # allow to see all states in plot





states = states_basemap.cx[-125:-50,25:50]



Base_states = states.plot(edgecolor= 'black', color = 'lightgrey')



data_simple_map = data_all[["Monitor_ID","State", "Latitude", "Longitude", "Observed_NO2_ppb"]]



data_simple_map_convert =  geopandas.GeoDataFrame(data_simple_map ,geometry=geopandas.points_from_xy(data_simple_map.Longitude, data_simple_map.Latitude))



Monitors_map = data_simple_map_convert['geometry']









Monitors_map_state = Monitors_map.plot(ax = Base_states, markersize = 15, color = "red" )

Monitors_map_state.set_ylabel('Longitude \N{DEGREE SIGN}N')

Monitors_map_state.set_xlabel('Longitude \N{DEGREE SIGN}E')

plt.savefig("Monitor_Map_US.png")


# Region 1



sns.set(style = "white", rc={'figure.figsize':(20,15)})





    

Monitors_Region_1 = states_basemap[(states_basemap["STUSPS"] == "CT") |  (states_basemap["STUSPS"] == "ME")|

                   (states_basemap["STUSPS"] == "MA") | (states_basemap["STUSPS"] == "NH") | 

                   (states_basemap["STUSPS"] == "RI") | (states_basemap["STUSPS"] == "VT")|

                                     (states_basemap["STUSPS"] == "NJ") | 

                   (states_basemap["STUSPS"] == "NY") | (states_basemap["STUSPS"] == "PA")]

                   

            





Base_Region_1 = Monitors_Region_1.plot(edgecolor= 'black', color = 'lightgrey' )





Monitors_map_R1 = data_simple_map_convert.plot(column= "Observed_NO2_ppb", ax = Base_Region_1,

                                               markersize = 20 ,cmap= 'rainbow', legend = True,

                                              legend_kwds = {'label': 'NO\u2082 concentration (ppb)', 

                                          'orientation':'horizontal',

                                                            'shrink': .75}

                                              )

                                              

                                        



Monitors_map_R1.set_xlim([-81,-67])

Monitors_map_R1.set_ylim([39,47.5])

Monitors_map_R1.set_ylabel('Latitude \N{DEGREE SIGN}N')

Monitors_map_R1.set_xlabel('Longitude \N{DEGREE SIGN}E')

plt.savefig("Monitor_Map_Region1_NE.png")
  #Region 2



sns.set(style = "white", rc={'figure.figsize':(20,15)})

    

Monitors_Region_2 = states_basemap[(states_basemap["STUSPS"] == "IL") |  (states_basemap["STUSPS"] == "IN")|

                   (states_basemap["STUSPS"] == "MI") | (states_basemap["STUSPS"] == "OH") | 

                    (states_basemap["STUSPS"] == "WI")|(states_basemap["STUSPS"] == "IA") |(states_basemap["STUSPS"] == "KS")|

                    (states_basemap["STUSPS"] == "MN") | (states_basemap["STUSPS"] == "MO") | 

                    (states_basemap["STUSPS"] == "NE") | (states_basemap["STUSPS"] == "ND") | 

                    (states_basemap["STUSPS"] == "SD") ]

                    





Base_Region_2 = Monitors_Region_2.plot(edgecolor= 'black', color = 'lightgrey' )





Monitors_map_R2 = data_simple_map_convert.plot(column= "Observed_NO2_ppb", ax = Base_Region_2,

                                               markersize = 20 ,cmap= 'rainbow', legend = True,

                                              legend_kwds = {'label': 'NO\u2082 concentration (ppb)', 

                                          'orientation':'horizontal',

                                                            'shrink': .75}

                                              )

                                        

                                            

Monitors_map_R2.set_xlim([-105,-80.4])

Monitors_map_R2.set_ylim([36.5,50])

Monitors_map_R2.set_ylabel('Latitude \N{DEGREE SIGN}N')

Monitors_map_R2.set_xlabel('Longitude \N{DEGREE SIGN}E')

plt.savefig("Monitor_Map_Region2_MW.png")
#Region 3

sns.set(style = "white", rc={'figure.figsize':(20,15)})

    

Monitors_Region_3 = states_basemap[(states_basemap["STUSPS"] == "DE") |  (states_basemap["STUSPS"] == "FL")|

                   (states_basemap["STUSPS"] == "GA") | (states_basemap["STUSPS"] == "MD") | 

                    (states_basemap["STUSPS"] == "NC")|(states_basemap["STUSPS"] == "SC") |  

                    (states_basemap["STUSPS"] == "VA")|(states_basemap["STUSPS"] == "DC") |

                    (states_basemap["STUSPS"] == "WV")| (states_basemap["STUSPS"] == "AR") |  

                    (states_basemap["STUSPS"] == "LA")|(states_basemap["STUSPS"] == "OK") |

                    (states_basemap["STUSPS"] == "TX")| (states_basemap["STUSPS"] == "AL") |  

                    (states_basemap["STUSPS"] == "KY")|(states_basemap["STUSPS"] == "MS") |

                    (states_basemap["STUSPS"] == "TN")]

                    

                    







Base_Region_3 = Monitors_Region_3.plot(edgecolor= 'black', color = 'lightgrey' )





Monitors_map_R3 = data_simple_map_convert.plot(column= "Observed_NO2_ppb", ax = Base_Region_3,

                                               markersize = 20 ,cmap= 'rainbow', legend = True,

                                              legend_kwds = {'label': 'Observed NO\u2082 concentration (ppb)', 

                                          'orientation':'horizontal',

                                                            'shrink': .75}

                                              )

                                        





Monitors_map_R3.set_xlim([-107,-73])

Monitors_map_R3.set_ylim([24,40.25])

Monitors_map_R3.set_ylabel('Latitude \N{DEGREE SIGN}N')

Monitors_map_R3.set_xlabel('Longitude \N{DEGREE SIGN}E')

plt.savefig("Monitor_Map_Region3_So.png")
# Region 4



sns.set(style = "white", rc={'figure.figsize':(20,15)})



Monitors_Region_4 = states_basemap[(states_basemap["STUSPS"] == "AZ") |  (states_basemap["STUSPS"] == "CO")|

                   (states_basemap["STUSPS"] == "ID") | (states_basemap["STUSPS"] == "MT") | 

                    (states_basemap["STUSPS"] == "NV")|(states_basemap["STUSPS"] == "NM") |  

                    (states_basemap["STUSPS"] == "UT")|(states_basemap["STUSPS"] == "WY")|

                  (states_basemap["STUSPS"] == "AK")|(states_basemap["STUSPS"] == "CA") |  

                    (states_basemap["STUSPS"] == "HI")|(states_basemap["STUSPS"] == "OR") |

                    (states_basemap["STUSPS"] == "WA")]





Base_Region_4 = Monitors_Region_4.plot(edgecolor= 'black', color = 'lightgrey' )





Monitors_map_R4 = data_simple_map_convert.plot(column= "Observed_NO2_ppb", ax = Base_Region_4,

                                               markersize = 14 ,cmap= 'rainbow', legend = True,

                                              legend_kwds = {'label': 'NO\u2082 concentration (ppb)', 

                                          'orientation':'horizontal',

                                                            'shrink': .5}

                                              )



Monitors_map_R4.set_xlim([-125,-102])

Monitors_map_R4.set_ylim([30,50])

Monitors_map_R4.set_ylabel('Latitude \N{DEGREE SIGN}N')

Monitors_map_R4.set_xlabel('Longitude \N{DEGREE SIGN}E')

plt.savefig("Monitor_Map_Region4_WT.png")
sns.set(style = "white", rc={'figure.figsize':(20,15)})



Monitors_CA = states_basemap[(states_basemap["STUSPS"] == "CA")]





Monitors_CA = Monitors_CA.cx[-125:-114,32:42]





Base_CA= Monitors_CA.plot(edgecolor= 'black', color = 'lightgrey' )





Monitors_map_CA = data_simple_map_convert.plot(column= "Observed_NO2_ppb", ax = Base_CA,

                                               markersize = 14 ,cmap= 'rainbow', legend = True,

                                              legend_kwds = {'label': 'NO\u2082 concentration (ppb)', 

                                          'orientation':'horizontal',

                                                            'shrink': .5}

                                              )







Monitors_map_CA.set_xlim([-125,-114])

Monitors_map_CA.set_ylim([32,42.5])

Monitors_map_CA.set_ylabel('Latitude \N{DEGREE SIGN}N')

Monitors_map_CA.set_xlabel('Longitude \N{DEGREE SIGN}E')

plt.savefig("Monitor_Map_Ca.png")


data_WRF = data_all.describe()



data_WRF = data_WRF.drop("count")



data_WRF = data_WRF.drop(columns = ["Latitude", "Longitude"], axis = 1)



print(data_WRF)
#all

corr_all = data_all.corr("pearson")



corr_all = corr_all["Observed_NO2_ppb"]

corr_all = corr_all.rename("All")





#R1

corr_R1 = data_R1.corr("pearson")





corr_R1 = corr_R1["Observed_NO2_ppb"]

corr_R1 = corr_R1.rename("Region 1")





#R2

corr_R2 = data_R2.corr("pearson")



corr_R2 = corr_R2["Observed_NO2_ppb"]

corr_R2 = corr_R2.rename("Region 2")



#R3



corr_R3 = data_R3.corr("pearson")



corr_R3 = corr_R3["Observed_NO2_ppb"]

corr_R3 = corr_R3.rename("Region 3")



#R4 





corr_R4 = data_R4.corr("pearson")



corr_R4 = corr_R4["Observed_NO2_ppb"]

corr_R4 = corr_R4.rename("Region 4")



df_corr = pd.concat([corr_all,corr_R1, corr_R2, corr_R3, corr_R4], axis = 1 )



df_corr = df_corr.reset_index()



df_corr = df_corr.dropna(axis=0, how="any")



df_corr = df_corr.drop(labels = [0,1,2,4,5], axis = 0)





df_corr = df_corr.reset_index(drop=True)



# Correlation for Impervious surfaces



df_corr_impervious = df_corr[1:23]



df_corr = df_corr.reset_index(drop=True)



print(df_corr)

df_corr_impervious = df_corr[1:23]



df_corr_impervious["within"] = [100,200,300,400,500, 600,700,800,1000,1200,1500,1800,2000,2500,3000,3500,4000,5000,6000,7000,8000,10000]





df_corr_impervious = df_corr_impervious.set_index("within")



sns.set(rc={'figure.figsize':(11.7,8.27)})

df_corr_impervious_plot = sns.scatterplot(data= df_corr_impervious, s=90)

plt.legend(loc='lower right',fontsize='x-large', title_fontsize='15')

df_corr_impervious_plot.set_ylabel('R-squared')

df_corr_impervious_plot.set_xlabel('Distance of buffer area used (m)')

plt.show()

plt.savefig("Correlation_impervious.png")
df_corr_population= df_corr[23:45]





df_corr_population["within"] = [100,200,300,400,500, 600,700,800,1000,1200,1500,1800,2000,2500,3000,3500,4000,5000,6000,7000,8000,10000]



df_corr_population = df_corr_population.set_index("within")





sns.set(rc={'figure.figsize':(11.7,8.27)})

df_corr_population_plot = sns.scatterplot(data= df_corr_population, s=90)

plt.legend(loc='lower right',fontsize='x-large', title_fontsize='15')

df_corr_population_plot.set_ylabel('R-squared')

df_corr_population_plot.set_xlabel('Distance of buffer area used (m)')

plt.show()

plt.savefig("Correlation_population.png")
df_corr_major= df_corr[45:67]



df_corr_major["within"] = [100,200,300,400,500, 600,700,800,1000,1200,1500,1800,2000,2500,3000,3500,4000,5000,6000,7000,8000,10000]



df_corr_major = df_corr_major.set_index("within")





sns.set(rc={'figure.figsize':(11.7,8.27)})

df_corr_major_plot = sns.scatterplot(data= df_corr_major, s=90)

plt.legend(loc='lower right',fontsize='x-large', title_fontsize='15')

df_corr_major_plot.set_ylabel('R-squared')

df_corr_major_plot.set_xlabel('Distance of buffer area used (m)')

plt.show()

plt.savefig("Correlation_majorroads.png")
df_corr_resident = df_corr[67:97]



df_corr_resident["within"] = [100,200,300,400,500, 600,700,800,1000,1200,1500,1800,2000,2500,3000,3500,4000,5000,6000,7000,8000,10000, 10500, 11000,11500,12000,12500,13000,13500,14000]



df_corr_resident = df_corr_resident.set_index("within")





sns.set(rc={'figure.figsize':(11.7,8.27)})

df_corr_resident_plot = sns.scatterplot(data= df_corr_resident, s=90)

plt.legend(loc='lower right',fontsize='x-large', title_fontsize='15')

df_corr_resident_plot.set_ylabel('R-squared')

df_corr_resident_plot.set_xlabel('Distance of buffer area used (m)')

plt.show()

plt.savefig("Correlation_residentroads.png")
df_corr_total = df_corr[97:]





df_corr_total["within"] = [100,200,300,400,500, 600,700,800,1000,1200,1500,1800,2000,2500,3000,3500,4000,5000,6000,7000,8000,10000, 10500, 11000,11500,12000,12500,13000,13500,14000]



df_corr_total = df_corr_total.set_index("within")





sns.set(rc={'figure.figsize':(11.7,8.27)})

df_corr_total_plot = sns.scatterplot(data= df_corr_total, s=90)

plt.legend(loc='lower right',fontsize='x-large', title_fontsize='15')

df_corr_total_plot.set_ylabel('R-squared')

df_corr_total_plot.set_xlabel('Distance of buffer area used (m)')

plt.show()

plt.savefig("Correlation_totalroads.png")