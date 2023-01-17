import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import rasterio as rio

import folium

import tifffile as tiff



def plot_points_on_map(dataframe,begin_index,end_index,latitude_column,latitude_value,longitude_column,longitude_value,zoom):

    df = dataframe[begin_index:end_index]

    location = [latitude_value,longitude_value]

    plot = folium.Map(location=location,zoom_start=zoom)

    color={ 'Hydro' : 'lightblue', 'Solar' : 'orange', 'Oil' : 'darkblue', 'Coal' : 'black', 'Gas' : 'lightgray', 'Wind' : 'green' }

    for i in range(0,len(df)):

        popup = folium.Popup(str(df.plant[i]))   #[i:i+1]))

        folium.Marker([df[latitude_column].iloc[i],df[longitude_column].iloc[i]],popup=popup, 

                      icon=folium.Icon(color=color[df.primary_fuel.iloc[i]])).add_to(plot)

    return(plot)



def overlay_image_on_puerto_rico(file_name,band_layer):

    band = rio.open(file_name).read(band_layer)

    m = folium.Map([lat, lon], zoom_start=8, width=500, height=400)

    folium.raster_layers.ImageOverlay(

        image=band,

        bounds = [[18.6,-67.3,],[17.9,-65.2]],

        colormap=lambda x: (1, 0, 0, x),

    ).add_to(m)

    return m



def plot_scaled(file_name):

    vmin, vmax = np.nanpercentile(file_name, (5,95))  # 5-95% stretch

    img_plt = plt.imshow(file_name, cmap='gray', vmin=vmin, vmax=vmax)

    plt.show()



def split_column_into_new_columns(dataframe,column_to_split,new_column_one,begin_column_one,end_column_one):

    for i in range(0, len(dataframe)):

        dataframe.loc[i, new_column_one] = dataframe.loc[i, column_to_split][begin_column_one:end_column_one]

    return dataframe
# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

power_plants = pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')

power_plants = split_column_into_new_columns(power_plants,'.geo','latitude',50,66)

power_plants = split_column_into_new_columns(power_plants,'.geo','longitude',31,48)

power_plants['latitude'] = power_plants['latitude'].astype(float)

a = np.array(power_plants['latitude'].values.tolist()) # 18 instead of 8

power_plants['latitude'] = np.where(a < 10, a+10, a).tolist() 


power_plants_df = power_plants.sort_values('capacity_mw',ascending=False).reset_index()

#power_plants_df['prod_hrs_p_day']=power_plants_df.estimated_generation_gwh*1000/(power_plants_df.capacity_mw*365)





bounds = [[18.6,-67.3,],[17.9,-65.2]]



power_plants_df['img_idx_lt']=(((18.6-power_plants_df.latitude)*148/(18.6-17.9))).astype(int)

power_plants_df['img_idx_lg']=((67.3+power_plants_df.longitude.astype(float))*475/(67.3-65.2)).astype(int)

power_plants_df['plant']=power_plants_df.name.str[:3]+power_plants_df.name.str[-1]+'_'+power_plants_df.primary_fuel

#power_plants_df['offset_img_aai']=(3-(power_plants_df.longitude.astype(float)+67)*2).astype(int)



power_plants=power_plants_df[['name','latitude','longitude','primary_fuel','capacity_mw','img_idx_lt','img_idx_lg','plant']]



power_plants
lat=18.200178; lon=-66.3 #-66.664513

plot_points_on_map(power_plants,0,425,'latitude',lat,'longitude',lon,9)

print('Total green (solar, wind, hydro) energy capacity in MW :', power_plants_df.loc[((power_plants_df['primary_fuel']=='Hydro') | (power_plants_df['primary_fuel']=='Solar') | (power_plants_df['primary_fuel']=='Wind'))

                    ,'capacity_mw'].sum())



print('Total gray (oil, gas, coal) energy capacity in MW :',power_plants_df.loc[((power_plants_df['primary_fuel']=='Coal') | (power_plants_df['primary_fuel']=='Oil') | (power_plants_df['primary_fuel']=='Gas'))

                    ,'capacity_mw'].sum())
import matplotlib.patches as mpatches



fig1 = plt.figure(figsize=(10, 5))



color={ 'Hydro' : 'lightblue', 'Solar' : 'orange', 'Oil' : 'darkblue', 'Coal' : 'black', 'Gas' : 'lightgray', 'Wind' : 'green' }

barcolor=[]

for fuel in power_plants_df.primary_fuel : barcolor.append(color[fuel]) 



fig1 = fig1.add_subplot(111)

fig1.bar(x=power_plants_df.index, height=power_plants_df.capacity_mw, width=0.6, color=barcolor)     

    

plt.yscale('log')

plt.title('Power plants in Puerto Rico by primary fuel and in descending order of capacity ')

plt.ylabel('Capacity (MW, log-scale)')

plt.xlabel('Powerplants in Puerto Rico')



patches=[]

for key, value in color.items(): patches.append(mpatches.Patch(color=value, label=key))

fig1.legend(handles=patches)



plt.show()
# add information on capacity, type of fuel and activity factor



# Information from eia.gov on electricity consumption of Puerto Rico gives a power consumption of 19.48 billion kWh (=19.480.000 MWh)for the year 2019

# Information from index.mundi.com on the fuel consumption of power generation in Puerte Rico gives a distribution of 40%/40%/18%/2% for oil/gas/coal/renewables



Prod_day=int(19480000/365)

print('Average emission factor per day (production in MWh/day) : ',Prod_day)





# With above information the drivers for the emission factor is calculated on a daily basis:



EF_oil=19480000*0.4/365   # MWh/day

EF_gas=19480000*0.4/365   # MWh/day

EF_coal=19480000*0.18/365  # MWh/day



# With the available capacity for oil, gas and coal plants the daily activity factor A is calculated



print('Emission factor (production in MWh/day) per day for oil: ',int(EF_oil),' gas: ',int(EF_gas),' and coal: ',int(EF_coal))



#print(gray.groupby(by='primary_fuel').capacity_mw.sum())



A_oil=EF_oil/power_plants_df.loc[power_plants_df.primary_fuel=='Oil','capacity_mw'].sum() 

A_gas=EF_gas/power_plants_df.loc[power_plants_df.primary_fuel=='Gas','capacity_mw'].sum() 

A_coal=EF_coal/power_plants_df.loc[power_plants_df.primary_fuel=='Coal','capacity_mw'].sum() 



print('Activity factor of power plants (average hrs/day) for oil, gas and coal ',A_oil/24,' gas: ',A_gas/24,' and coal: ',A_coal/24)





# inspection of image information

image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180708T172237_20180714T190743.tif'

img=rio.open(image)



# print('Shape of array with data points :',tiff.imread(image).shape)

# img.descriptions

from datetime import datetime



files=[]

for dirname, _, filenames in os.walk('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))



# read all the absorbing aerosol index data into one list of arrays

aai_first_day=[]

aai_first_key=[]

aai_last_day=[]

aai_arr=[]

band=5 #aerosol index

#band=6 # cloud fraction

for i in range(0,len(files)):

    aai_first_day.append(datetime.strptime(files[i][76:91], '%Y%m%dT%H%M%S').date())

    aai_first_key.append(datetime.strptime(files[i][76:91], '%Y%m%dT%H%M%S').toordinal()+1) # correction of + 1 day in order to sync on climate data

    aai_last_day.append(datetime.strptime(files[i][92:107], '%Y%m%dT%H%M%S').date())

    #aai_first_day.append(pd.Timestamp(files[i][76:91]))

    #aai_last_day.append(pd.Timestamp(files[i][92:107]))

    aai_arr.append(rio.open(files[i]).read(band+1))



from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()





# a=[]

# for i,arr in enumerate(aai_arr): a.append(np.nanmean(arr))

a=[]

a_pos=[]

nll=[]

for i in range(0,len(aai_arr)): 

    a.append(np.nanmean(aai_arr[i]))

    a_pos.append(np.nanmean(np.clip(aai_arr[i],0,10000)))

    nll.append(pd.isnull(aai_arr[i]).sum().sum())



#aai_rgn=pd.DataFrame({'first': aai_first_day,'last':aai_last_day,'aai_rgn' : a, 'nll' : nll })

aai_rgn=pd.DataFrame({ 'first': aai_first_day,'last':aai_last_day,'aai_rgn' : a_pos, 'nll' : nll, 'aai_raw' : a,'key_date' : aai_first_key })

aai_rgn=aai_rgn.sort_values('first')

aai_rgn=aai_rgn.reset_index()



fig1 = plt.figure(figsize=(20, 10))

fig1.suptitle("data cleaning - mean of aai and number of nan per observation for 1) raw data (upper graphs) 2) cleaned data, #nan < 5% (middle graphs) 3) clipped data (lower graphs)")

ax1 = fig1.add_subplot(321)

ax1.plot(aai_rgn.iloc[:,1], aai_rgn.iloc[:,5], label='average aai per day - raw data', color='b')

ax1.legend()

ax2 = fig1.add_subplot(322)

ax2.plot(aai_rgn.iloc[:,1], aai_rgn.iloc[:,4], label='# nan per observation', color='b')

ax2.legend()



aai_rgn=aai_rgn.loc[aai_rgn.nll <3515,:] # only select observations with # nan < 5%



ax3 = fig1.add_subplot(323)

ax3.plot(aai_rgn.iloc[:,1], aai_rgn.iloc[:,5], label='average aai per day - cleaned for data with nan > 5%', color='b')

ax3.legend()

ax4 = fig1.add_subplot(324)

ax4.plot(aai_rgn.iloc[:,1], aai_rgn.iloc[:,4], label='# nan per observation', color='b')

ax4.legend()



aai_rgn=aai_rgn.loc[aai_rgn.nll <3515,:] # only select observations with # nan < 5%



ax5 = fig1.add_subplot(325)

ax5.plot(aai_rgn.iloc[:,1], aai_rgn.iloc[:,3], label='average aai per day - data <0 clipped to 0', color='b') 

ax5.legend()

ax6 = fig1.add_subplot(326)

ax6.plot(aai_rgn.iloc[:,1], aai_rgn.iloc[:,4], label='# nan per observation', color='b')

ax6.legend()
# read only the absorbing aerosol index arrays with a nan-percentage <5% into one list of arrays for calculation of local aai data

files=[]

for dirname, _, filenames in os.walk('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))



aai_first_day=[]

aai_last_day=[]

aai_first_key=[]

aai_arr=[]

#aai_arr_pos=[]

band=5 #aerosol index

#band=6 # cloud fraction

for i in range(0,len(files)):

    a=rio.open(files[i]).read(band+1)

    if pd.isnull(a).sum().sum() < 3515:

        aai_first_day.append(datetime.strptime(files[i][76:91], '%Y%m%dT%H%M%S').date())

        aai_first_key.append(datetime.strptime(files[i][76:91], '%Y%m%dT%H%M%S').toordinal()+1) # correction of + 1 day in order to sync on climate data

        aai_last_day.append(datetime.strptime(files[i][92:107], '%Y%m%dT%H%M%S').date())        

        aai_arr.append(np.clip(a,0,10000))  # clip negative values to zero

#        aai_arr.append(a)  # raw values

        

# aai_arr is list of arrays with cleaned aai values (negative values clipped to zero, # nan in array < 5%)

# aai_rgn is list of mean values of aai for the whole region (based on cleaned data)

#print(len(aai_rgn))

#print(len(aai_arr))
# inspection of measurements with highest aai



aai_rgn.loc[aai_rgn.aai_rgn>.8,['key_date','first','last','aai_rgn']]
image1 = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180711T162527_20180718T185658.tif'

image2 = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180816T164847_20180822T182145.tif'



# image1 : aai_rgn>0.8

# image2 : aai_rgn approx 0.01



#inspection of aai



image=image1

img=rio.open(image)



band=5

print('inspection of image with high absorbing_aerosol_index (average aai for the region > 0.8)') #, img.descriptions[band])



image_band = rio.open(image).read(band+1)



#plot_scaled(image_band)



f2 = folium.Figure(width=500, height=400, title=img.descriptions[band])

m = folium.Map([lat, lon], min_zoom=8, max_zoom=8, width='100%', height='100%').add_to(f2)

folium.raster_layers.ImageOverlay(

    image=image_band,

    bounds = [[18.6,-67.3,],[17.9,-65.2]],

    colormap=lambda x: (1, 0, 0, x),

).add_to(m)

f2
gray=power_plants.loc[((power_plants['primary_fuel']=='Coal') | (power_plants['primary_fuel']=='Oil') | (power_plants['primary_fuel']=='Gas')),

                         ['name','primary_fuel','capacity_mw','img_idx_lt','img_idx_lg','plant']]

gray.head()
# defining mask for locations with gray powerplants

locations=np.zeros((148,475))



# setting the geographical area (+/- n pixels) around a plant location for averaging local data

#n=10

n=11 #maximum value to stay within bounds of image



for j in range(0,len(gray)):

    locations[gray.iloc[j,3]-n:gray.iloc[j,3]+n,gray.iloc[j,4]-n:gray.iloc[j,4]+n]=np.ones((2*n,2*n))



#plot_scaled(locations)

print('Overview of areas on Puerto Rico with power plants running on oil, gas or coal. These areas are selected for modelling aai vs emission factors')



f1 = folium.Figure(width=500, height=400)

m = folium.Map([lat, lon], min_zoom=8, max_zoom=8, width='100%', height='100%').add_to(f1)  #zoom_start=8

folium.raster_layers.ImageOverlay(

    image=locations,

    bounds = [[18.6,-67.3,],[17.9,-65.2]],

    colormap=lambda x: (1, 0, 0, x),

).add_to(m)

f1
# aai value in proximity of all plants with all locations in location mask - proximity is +/- n points from location of plant



aai=[]

for j in range(0,len(gray)):

    idx_lt=gray.iloc[j,3]

    idx_lg=gray.iloc[j,4]

    

    aai_j=[]

    for i in range(0,len(aai_arr)):

        aai_j.append(np.nanmean(aai_arr[i][idx_lt-n:idx_lt+n,idx_lg-n:idx_lg+n])) # calculate average of aai for location of plant

    

    aai.append(aai_j)

    


aa=pd.DataFrame({'key_date':np.array(aai_first_key), 'first': aai_first_day,'last':aai_last_day}) 



for j in range(0,len(gray)):

    aa[gray.iloc[j,5]]=aai[j]  #add average of aai for location of plant to dataframe with column name from df gray.plant



print('size of dataframe with aai data for gray-energy power-plant locations: ',aa.shape)



# sorting dataframe on date to produce ordered time series

aa=aa.sort_values('key_date')

aa=aa.reset_index()

aa=aa.drop(columns=['index'])

aa=aa.fillna(0)

aa.head()



fig3 = plt.figure(figsize=(20, 10))

fig3.suptitle("absorbing aerosol index as function of time for different power plants")

#fig3 = fig3.add_subplot(111)

offset=0

#offset=8



ax=[]

for i in range(0,8):

    ax.append(fig3.add_subplot(421+i))

    ax[i].plot(aai_rgn.iloc[:,1], aai_rgn.iloc[:,3], label='mean_region', color='r')

    ax[i].plot(aa.iloc[:,0], aa.iloc[:,3+i+offset], label=aa.columns[3+i+offset], color='b')

    ax[i].set(ylim=(0, 2))                       #xlim=(-3, 3), ylim=(-3, 3))

    ax[i].set_xlabel('time')

    ax[i].set_ylabel('absorbing aerosol index')

#    ax[i].set_title("aerosol index as function of time")

    ax[i].legend()





fig4 = plt.figure(figsize=(20, 10))

fig4.suptitle("absorbing aerosol index as function of time for different power plants")

#fig3 = fig3.add_subplot(111)

#offset=0

offset=8



ax2=[]

for i in range(0,8):

    ax2.append(fig4.add_subplot(421+i))

    ax2[i].plot(aai_rgn.iloc[:,1], aai_rgn.iloc[:,3], label='mean_region', color='r')

    ax2[i].plot(aa.iloc[:,1], aa.iloc[:,3+i+offset], label=aa.columns[3+i+offset], color='b')

    ax2[i].set(ylim=(0, 2))                       #xlim=(-3, 3), ylim=(-3, 3))

    ax2[i].set_xlabel('time')

    ax2[i].set_ylabel('absorbing aerosol index')

#    ax2[i].set_title("aerosol index as function of time")

    ax2[i].legend()
#simplified emissions-factor as the average aai of the plant location divided by the capacity of the plant  



aai_plant=aa.drop(columns=['key_date','first','last']).mean()



print('yearly average aai for the whole region : ',aai_rgn.aai_rgn.mean())



fig5 = plt.figure(figsize=(20, 5))

fig5.suptitle("yearly average emission per plant location")

ax5 = fig5.add_subplot(111)

ax5.plot(aai_plant.index, aai_plant.values, label='average per plant', color='b')

ax5.plot(aai_plant.index, np.ones((len(aai_plant)))*aai_rgn.aai_rgn.mean(), label='average for the region', color='r')

ax5.legend()

# simplified_emissions_factor = float(average_no2_emission/quantity_of_electricity_generated)

# print('Simplified emissions factor (S.E.F.) for a single power plant on the island of Vieques =  \n\n', simplified_emissions_factor, 'S.E.F. units')
# image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gldas/gldas_20180702_1500.tif'

# img1=rio.open(image)

# image_band = rio.open(image).read(3)

# print(img1.descriptions)

# plot_scaled(image_band)



# # for i in range(1,13):

# #     image_band = rio.open(image).read(i)

# #     print(img1.descriptions[i-1])

# #     plot_scaled(image_band)



# # image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2018072118.tif'

# # img2=rio.open(image)

# # image_band = rio.open(image).read(3)

# # print(img2.descriptions)

# # plot_scaled(image_band)



# #overlay_image_on_puerto_rico(image,band_layer=3)
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2018072118.tif'

img2=rio.open(image)

print('Available information on climate factors')



for i in range(1,7):

    image_band = rio.open(image).read(i)

    print(img2.descriptions[i-1])

    plot_scaled(image_band)

files=[]

for dirname, _, filenames in os.walk('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))



# read all the climate data into lists of arrays

gfs_day=[]

gfs_key=[]

temp_arr=[]

spec_hum_arr=[]

rel_hum_arr=[]

u_wind_arr=[]

v_wind_arr=[]

pr_water_arr=[]

#band=0 #temperature_2m_above_ground

#band=6 # cloud fraction

for i in range(0,len(files)):

    gfs_day.append(datetime.strptime(files[i][68:78], '%Y%m%d%H').date())

    gfs_key.append(datetime.strptime(files[i][68:78], '%Y%m%d%H').toordinal())

    temp_arr.append(rio.open(files[i]).read(1)) #temperature_2m_above_ground

    spec_hum_arr.append(rio.open(files[i]).read(2)) #specific_humidity_2m_above_ground

    rel_hum_arr.append(rio.open(files[i]).read(3)) # relative_humidity_2m_above_ground

    u_wind_arr.append(rio.open(files[i]).read(4)) # u_component_of_wind_10m_above_ground

    v_wind_arr.append(rio.open(files[i]).read(5)) # v_component_of_wind_10m_above_ground

    pr_water_arr.append(rio.open(files[i]).read(6)) # precipitable_water_entire_atmosphere

    
#gfs data is clean - geen nan in data!



t=[] ; s=[] ; r=[] ; u=[] ; v=[] ; p=[]



for i in range(0,len(temp_arr)): 

    t.append(np.nanmean(temp_arr[i]))

    s.append(np.nanmean(spec_hum_arr[i]))

    r.append(np.nanmean(rel_hum_arr[i]))

    u.append(np.nanmean(u_wind_arr[i]))

    v.append(np.nanmean(v_wind_arr[i]))

    p.append(np.nanmean(pr_water_arr[i]))



gfs_rgn=pd.DataFrame({'day': gfs_day,'temp' : t, 'spec_hum' : s, 'rel_hum' : r, 'u_wind' : u, 'v_wind' : v, 'pr_water' : p, 'gfs_key': gfs_key })

gfs_rgn=gfs_rgn.sort_values('day')

gfs_rgn=gfs_rgn.reset_index()



fig10 = plt.figure(figsize=(20, 10))

fig10.suptitle("Average values for the region: temperature, specific_humidity, relative_humidity, u_comp_of_wind, v_comp_of_wind, precipitable_water (all left axis), aai on right axis")

ax1 = fig10.add_subplot(321)

ax1.plot(gfs_rgn.iloc[:,1], gfs_rgn.iloc[:,2], label='average temperature', color='b')

ax12 = ax1.twinx()

ax12.plot(aai_rgn.iloc[:,1], aai_rgn.iloc[:,3], label='average aai per day ', color='r') # aai_rgn.iloc[:,5] : waarde 3 geeft alleen positieve waardes van aai

ax1.legend() ; ax12.legend()

ax2 = fig10.add_subplot(322)

ax2.plot(gfs_rgn.iloc[:,1], gfs_rgn.iloc[:,3], label='average specific_humidity', color='b')

ax22 = ax2.twinx()

ax22.plot(aai_rgn.iloc[:,1], aai_rgn.iloc[:,3], label='average aai per day ', color='r')

ax2.legend() ; ax22.legend()

ax3 = fig10.add_subplot(323)

ax3.plot(gfs_rgn.iloc[:,1], gfs_rgn.iloc[:,4], label='average relative_humidity', color='b')

ax32 = ax3.twinx()

ax32.plot(aai_rgn.iloc[:,1], aai_rgn.iloc[:,3], label='average aai per day ', color='r')

ax3.legend() ; ax32.legend()

ax4 = fig10.add_subplot(324)

ax4.plot(gfs_rgn.iloc[:,1], gfs_rgn.iloc[:,5], label='average u_comp_of_wind', color='b')

ax42 = ax4.twinx()

ax42.plot(aai_rgn.iloc[:,1], aai_rgn.iloc[:,3], label='average aai per day ', color='r')

ax4.legend() ; ax42.legend()

ax5 = fig10.add_subplot(325)

ax5.plot(gfs_rgn.iloc[:,1], gfs_rgn.iloc[:,6], label='average v_comp_of_wind', color='b')

ax52 = ax5.twinx()

ax52.plot(aai_rgn.iloc[:,1], aai_rgn.iloc[:,3], label='average aai per day ', color='r')

ax5.legend() ; ax52.legend()

ax6 = fig10.add_subplot(326)

ax6.plot(gfs_rgn.iloc[:,1], gfs_rgn.iloc[:,7], label='average precipitable_water', color='b')

ax62 = ax6.twinx()

ax62.plot(aai_rgn.iloc[:,1], aai_rgn.iloc[:,3], label='average aai per day ', color='r')

ax6.legend() ; ax62.legend()
# climate values in proximity of all plants with all locations in location mask - proximity is +/- n points from location of plant



temp=[]

spec_hum=[]

rel_hum=[]

u_wind=[]

v_wind=[]

pr_water=[]

for j in range(0,len(gray)):

    idx_lt=gray.iloc[j,3]

    idx_lg=gray.iloc[j,4]

    

    temp_j=[] ; rel_hum_j=[] ; spec_hum_j=[] ; u_wind_j=[] ; v_wind_j=[] ; pr_water_j=[]

    for i in range(0,len(temp_arr)):

        temp_j.append(np.nanmean(temp_arr[i][idx_lt-n:idx_lt+n,idx_lg-n:idx_lg+n])) # calculate average of temp for location of plant

        spec_hum_j.append(np.nanmean(spec_hum_arr[i][idx_lt-n:idx_lt+n,idx_lg-n:idx_lg+n]))

        rel_hum_j.append(np.nanmean(rel_hum_arr[i][idx_lt-n:idx_lt+n,idx_lg-n:idx_lg+n]))

        u_wind_j.append(np.nanmean(u_wind_arr[i][idx_lt-n:idx_lt+n,idx_lg-n:idx_lg+n]))

        v_wind_j.append(np.nanmean(v_wind_arr[i][idx_lt-n:idx_lt+n,idx_lg-n:idx_lg+n]))

        pr_water_j.append(np.nanmean(pr_water_arr[i][idx_lt-n:idx_lt+n,idx_lg-n:idx_lg+n]))

    temp.append(temp_j)

    spec_hum.append(spec_hum_j)

    rel_hum.append(rel_hum_j)

    u_wind.append(u_wind_j)

    v_wind.append(v_wind_j)

    pr_water.append(pr_water_j)

    
gray=gray.iloc[:,:6]



# weight of each powerplant as input to the emission model is the same. The model will calculate the relative weights of each plant.

gray.loc[:,'EF_wght']=1



# aggregation of climate data per plant location into one dataframe, addition of aai data per plant location 

# only use data for the dates that coincide for aai-data and for climate data



ww=pd.DataFrame({'key_date':gfs_key})



XX=pd.DataFrame({})

for j in range(0,len(gray)):

    #ww[gray.iloc[j,5]]=temp[j]  #add average of aai for location of plant to dataframe with column name from df gray.plant

    ww['temp']=temp[j]

    ww['spec_hum']=spec_hum[j]

    ww['rel_hum']=rel_hum[j]

    ww['u_wind']=u_wind[j]

    ww['v_wind']=v_wind[j]

    ww['pr_water']=pr_water[j]

        

    x=ww.groupby(by='key_date').agg(['max','min','mean'])

    

    X=pd.merge(aa.loc[:,['key_date',gray.iloc[j,5]]], x, how='inner', on='key_date')

    X=X.rename(columns = {gray.iloc[j,5]:'aai'})

    

    c=gray.iloc[j,5]   #'EF_'+gray.iloc[j,5]

    X[c]=np.ones((len(X)))*gray.iloc[j,6] # addition of EF_wght for each plant to the dataframe

    

    XX=pd.concat([XX,X], axis=0, sort=False) # aggregation of dataframe per plantlocation into one dataframe



XX=XX.fillna(0) 

XX=XX.reset_index()

XX    

# i=13 #5 #13 #0-13

# j=13 #5 #13 #0-13

# offset=0

# fig12 = plt.figure(figsize=(20, 5))

# fig12.suptitle("Average values for the region: temperature, specific_humidity, relative_humidity, u_comp_of_wind, v_comp_of_wind, precipitable_water")

# ax1 = fig12.add_subplot(211)

# ax1.plot(XX.iloc[i*324:(i+1)*324,1], XX.iloc[i*324:(i+1)*324,18], label=XX.columns[20], color='b')

# ax12 = ax1.twinx()

# ax12.plot(XX.iloc[i*324:(i+1)*324,1], XX.iloc[i*324:(i+1)*324,2], label=XX.columns[23+i], color='r') # aai_rgn.iloc[:,5] : waarde 3 geeft alleen positieve waardes van aai

# ax1.legend() ; ax12.legend()

# ax2 = fig12.add_subplot(212)

# ax2.plot(XX.iloc[j*324:(j+1)*324,1], XX.iloc[j*324:(j+1)*324,18], label=XX.columns[20], color='b')

# ax22 = ax2.twinx()

# ax22.plot(XX.iloc[j*324+offset:(j+1)*324+offset,1], XX.iloc[j*324:(j+1)*324,2], label=XX.columns[23+i], color='r')

# ax2.legend() ; ax22.legend()


y=XX['aai']



X=XX.drop(columns=['index','key_date','aai'])

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

import xgboost as xgb



max_depth = 3

min_child_weight = 10

subsample = 0.5

colsample_bytree = 0.6

objective = "reg:squarederror" #'reg:linear',#"reg:squarederror"

num_estimators = 500 #2000 #1000 #3000  #200

learning_rate =  0.01 #0.01  #0.05 #0.003 # 0.3



xgb_reg = xgb.XGBRegressor(max_depth=max_depth,

            min_child_weight=min_child_weight,

            subsample=subsample,

            colsample_bytree=colsample_bytree,

            objective=objective,

            n_estimators=num_estimators,

            learning_rate=learning_rate,

            early_stopping_rounds=100,

            num_boost_round = 2000)



kf = KFold(n_splits=5, random_state=42, shuffle=True) # n_splits was 5





i=0

testscore=[]

feature_imp=pd.DataFrame({'feature': X.columns})

for train_index, test_index in kf.split(X, y):

#for train_index, test_index in gkf.split(X, y, groups):

    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]

    y_train, y_test = y[train_index], y[test_index]

                      

    xgb_reg.fit(X_train, np.ravel(y_train)) 



    y_pred = xgb_reg.predict(X_test)

    test_score1 = mean_squared_error(y_test, y_pred)  

    

    testscore.append(test_score1)

    i=i+1

    feature_imp['importance'+str(i)]=xgb_reg.feature_importances_



feature_imp['mean']=feature_imp.iloc[:,1:i].mean(axis=1)

feature_imp['std']=feature_imp.iloc[:,1:i].std(axis=1)    

print('mean_squared_error on test_set:', testscore, np.mean(testscore))
fig12 = plt.figure(figsize=(20, 5))

fig12.suptitle("visual inspection of prediction of aai (y_pred) vs measured value of aai (y_test)")

ax1 = fig12.add_subplot(111)

ax1.plot(range(0,len(y_test)), y_test, label='y_test', color='b')

#ax12 = ax1.twinx()

ax1.plot(range(0,len(y_test)), y_pred, label='y_pred', color='r') # aai_rgn.iloc[:,5] : waarde 3 geeft alleen positieve waardes van aai

ax1.legend() #; ax12.legend()
feature_imp=feature_imp.sort_values('mean', ascending=False)

#feature_imp
feature_imp=feature_imp.sort_values('mean', ascending=True)



plt.figure(figsize=(16, 12))

plt.title("Feature importances in emission model of power plants in Puerto Rico")

plt.barh(range(X.shape[1]), feature_imp['mean'],

       color="r", xerr=feature_imp['std'], align="center")

# If you want to define your own labels,

# change indices to a list of labels on the following line.

plt.yticks(range(X.shape[1]), feature_imp['feature'])

plt.ylim([-1, X.shape[1]])

plt.show()
gray=gray.iloc[:16,:7]



prod_features=feature_imp.rename(columns= {'feature':'plant'})

gray=pd.merge(gray, prod_features.loc[:,['plant','mean']], how='left', on='plant')

gray=gray.rename(columns= {'mean':'emission_contrib'}) # emission contribution as calculated from the feature importances of the model

#gray
print('Contribution of production factors in the model to the measured total emissions (%) :', gray.emission_contrib.sum()*100)



gray=gray.sort_values('emission_contrib', ascending=False)



# calculation of maximum daily production (MWh) based on max. capacity of power plants

gray['EF_max_MWh_day']=(gray['capacity_mw']*24).astype(int)



# daily energy production for the region (MWh) distributed to plants according to emission distribution from the model

gray['hist_emission_MWh_day']=(gray['emission_contrib']*Prod_day/gray.emission_contrib.sum()).astype(int)



# activity factor calculated from model emission distribution and maximum daily production

gray['Activity_%']=(gray['hist_emission_MWh_day']*100/gray['EF_max_MWh_day']).clip(upper=100).astype(int)



# pollution factor: if activity > 100 (%) then emissions cannot be explained by power production because the plant is running beyond maximum capacity.

# the pollution factor can explain these additional emissions attributable to choice of primary_fuel and generation of technology (age of plant).

gray['Pollution_factor']=(gray['hist_emission_MWh_day']/gray['EF_max_MWh_day']).astype(int)

gray
# fuel distribution purely based on historical emissions

print('fuel distribution based on historical emissions')

print(gray.groupby(by='primary_fuel').hist_emission_MWh_day.sum()/gray.hist_emission_MWh_day.sum())

print('  ')

# fuel distribution based on historical emissions with capacity restrictions used where applicable. 

gray['est_prod']=gray['Activity_%']*gray['EF_max_MWh_day']



print('fuel distribution based on historical emissions with capacity restrictions')

print(gray.groupby(by='primary_fuel').est_prod.sum()/gray.est_prod.sum())
# from kaggle_secrets import UserSecretsClient

# from google.oauth2.credentials import Credentials

# import ee

# import folium



# def add_ee_layer(self, ee_image_object, vis_params, name):

#   # https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb

#   map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)

#   folium.raster_layers.TileLayer(

#     tiles = map_id_dict['tile_fetcher'].url_format,

#     attr = 'Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',

#     name = name,

#     overlay = True,

#     control = True

#   ).add_to(self)



# def plot_ee_data_on_map(dataset,column,begin_date,end_date,minimum_value,maximum_value,latitude,longitude,zoom):

#     # https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb

#     folium.Map.add_ee_layer = add_ee_layer

#     vis_params = {

#       'min': minimum_value,

#       'max': maximum_value,

#       'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}

#     my_map = folium.Map(location=[latitude,longitude], zoom_start=zoom, height=500)

#     s5p = ee.ImageCollection(dataset).filterDate(

#         begin_date, end_date)

#     my_map.add_ee_layer(s5p.first().select(column), vis_params, 'Color')

#     my_map.add_child(folium.LayerControl())

#     display(my_map)
#!cat ~/.config/earthengine/credentials
# user_secret = "earth_engine" # Your user secret, defined in the add-on menu of the notebook editor

# refresh_token = UserSecretsClient().get_secret(user_secret)

# credentials = Credentials(

#         None,

#         refresh_token=refresh_token,

#         token_uri=ee.oauth.TOKEN_URI,

#         client_id=ee.oauth.CLIENT_ID,

#         client_secret=ee.oauth.CLIENT_SECRET,

#         scopes=ee.oauth.SCOPES)

# ee.Initialize(credentials=credentials)
# dataset = "COPERNICUS/S5P/NRTI/L3_NO2"

# column = 'absorbing_aerosol_index'

# begin_date = '2018-07-08'

# end_date = '2018-07-14'

# minimum_value = 0.1 # 0.0000000001

# maximum_value = 0.4 # 1

# latitude = 60.17

# longitude = 25.94

# zoom = 5

# plot_ee_data_on_map(dataset,column,begin_date,end_date,minimum_value,maximum_value,latitude,longitude,zoom)

# dataset = "NOAA/GFS0P25"

# column = 'temperature_2m_above_ground'

# begin_date = '2018-07-08'

# end_date = '2018-07-14'

# minimum_value = 0

# maximum_value = 50

# latitude = 18.20

# longitude = -66.66

# zoom = 8

# plot_ee_data_on_map(dataset,column,begin_date,end_date,minimum_value,maximum_value,latitude,longitude,zoom)



# dataset = "NASA/GLDAS/V021/NOAH/G025/T3H"

# column = 'Tair_f_inst'

# begin_date = '2018-07-08'

# end_date = '2018-07-14'

# minimum_value = 270

# maximum_value = 310

# latitude = 18.20

# longitude = -66.66

# zoom = 8

# plot_ee_data_on_map(dataset,column,begin_date,end_date,minimum_value,maximum_value,latitude,longitude,zoom)