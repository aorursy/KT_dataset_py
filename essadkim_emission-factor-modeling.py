# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from matplotlib import animation, rc
import datetime
import rasterio as rio
import folium
#from scipy.ndimage import gaussian_filter
from utils_ds4g import *
from no2_solver import *
sys.path.append('')
main_folder = '/kaggle/input/ds4g-environmental-insights-explorer/'
folder_s5p_no2 = main_folder+'eie_data/s5p_no2/'

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import sys
sys.path.append('/kaggle/input/ds4g-environmental-insights-explorer/')





# Any results you write to the current directory are saved as output.
# Important parameter
shape0 = (148, 475)
lat_limits = np.array([ 17.9, 18.56])
lon_limits = np.array([-67.32, -65.19])
[lat, lon] = [lat_limits.mean(),  lon_limits.mean()]
Rt = 6371*1.e3
T_year_h = 24*365.25
T_day_s = 24*3600

T_year_s = T_year_h*3600
# convert the polor coordiate to a local cartesian coordinate by assuming a locally flat area 
X = Rt*np.cos(lat*np.pi/180)*np.linspace(np.pi/180*lon_limits[0],np.pi/180*lon_limits[1],shape0[1])
Y = Rt*np.linspace(np.pi/180*lat_limits[0],np.pi/180*lat_limits[1],shape0[0])
dx = np.abs(X[1]-X[0])
dy = np.abs(Y[1]-Y[0])

power_plants = pd.read_csv(main_folder+'eie_data/gppd/gppd_120_pr.csv')
power_plants.describe()
# drop useless columns
for y in range(2013,2018):
    del power_plants['generation_gwh_'+str(y)]
for i in range(1,4):
    del power_plants['other_fuel'+str(i)]
del power_plants['country']
del power_plants['country_long']
power_plants.head()
# fuel mix 
print(power_plants.primary_fuel.value_counts())
sns.countplot(power_plants.primary_fuel)

# the total capacity of power plant by fuel type


power_plants.groupby('primary_fuel',group_keys=True).sum().reset_index().plot.bar(x='primary_fuel',y='capacity_mw')

thermal_power_plants = power_plants[power_plants.primary_fuel.isin(['Coal','Gas','Oil'])].reset_index()
print('percentage of fossil fuel power plant capacity',thermal_power_plants.capacity_mw.sum()/power_plants.capacity_mw.sum())
# capacity factor = total generated energy in one year/ (maximal energy that can be generated over the year)

thermal_power_plants['capacity_factor'] = 1.e3*thermal_power_plants['estimated_generation_gwh']/ (T_year_h*thermal_power_plants['capacity_mw'])
thermal_power_plants.plot.bar(x='primary_fuel',y='capacity_factor')

#thermal_power_plants.loc[thermal_power_plants.primary_fuel=='Coal'].capacity_factor  = 0.5
thermal_power_plants.at[14,'capacity_factor'] = 0.5 # we update the capacity factor of the coal power plant
thermal_power_plants['estimated_generation_gwh'] =  (T_year_h*thermal_power_plants['capacity_mw'])*thermal_power_plants['capacity_factor']/(1.e3) 
thermal_power_plants.plot.bar(x='primary_fuel',y='capacity_factor')

thermal_power_plants.groupby('primary_fuel',group_keys=True).sum().reset_index().plot.bar(x='primary_fuel',y='estimated_generation_gwh',title='total generated electricty per fuel')

thermal_power_plants = split_column_into_new_columns(thermal_power_plants,'.geo','latitude',50,66)
thermal_power_plants = split_column_into_new_columns(thermal_power_plants,'.geo','longitude',31,48)
thermal_power_plants['latitude'] = thermal_power_plants['latitude'].astype(float)
a = np.array(thermal_power_plants['latitude'].values.tolist()) # 18 instead of 8
thermal_power_plants['latitude'] = np.where(a < 10, a+10, a).tolist() 


plot_points_on_map(thermal_power_plants,0,425,'latitude',lat,'longitude',lon,9)
# the monthly capacity factor are in https://www.eia.gov/electricity/data/browser/#/topic/0?agg=2,0,1&fuel=vtvv&geo=g&sec=g&freq=M&start=201801&end=201912&ctype=linechart&ltype=pin&rtype=s&maptype=0&rse=0&pin=
monthly_generated_electricity_USA = pd.read_csv('/kaggle/input/monthly-generated-electricity/Net_generation_for_all_sectors.csv')
monthly_generated_electricity_USA.set_index('description',inplace=True)
monthly_generated_electricity_USA = monthly_generated_electricity_USA.transpose().reset_index()
monthly_generated_electricity_USA = monthly_generated_electricity_USA.rename(columns={"index": "date"})
monthly_generated_electricity_USA.head()
month_num_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12 }

monthly_generated_electricity_USA['month'] = monthly_generated_electricity_USA['date'].apply(lambda r: r.split(' ')[0])
monthly_generated_electricity_USA['month_num'] = monthly_generated_electricity_USA['month'].apply(lambda r: month_num_map[r])

u = monthly_generated_electricity_USA.groupby('month',group_keys=True).mean()
u.sort_values(by='month_num',inplace=True)
u = u.reset_index()

u['all_fuels_month_percent'] = u['all fuels']/(u['all fuels'].sum())
fig,ax = plt.subplots(1,2,figsize=(14,6))
u.drop(['month_num'],axis=1)
u.plot(x='month',y='all fuels',ax=ax[0])
ax[0].set_ylabel('monthly generated electricity (gwh)')
u.plot(x='month',y='all_fuels_month_percent',ax=ax[1])
ax[1].set_ylabel('month fraction of generated electricity')

for [month, fraction] in zip(u.month,u.all_fuels_month_percent):
    thermal_power_plants['estimated_generation_gwh_'+month] = thermal_power_plants['estimated_generation_gwh']*fraction

molar_mass_NO2 = 46.005 # g/mol
MBTU_to_KWH = 0.29307
Emission_factor_g_per_MBTU = {'Coal':1.6,'Gas':0.1,'Oil':0.6}
Emission_factor_mol_per_kwh = Emission_factor_g_per_MBTU
for k in Emission_factor_g_per_MBTU.keys():
    Emission_factor_mol_per_kwh[k] = Emission_factor_g_per_MBTU[k]/molar_mass_NO2*MBTU_to_KWH
    
thermal_power_plants['estimated_emission_factor_mol_kwh'] = thermal_power_plants.primary_fuel.apply(lambda r:Emission_factor_mol_per_kwh[r])
for [m, p] in zip(u.month,u.all_fuels_month_percent):
    thermal_power_plants['estimated_emission_mol_'+m] = 1.e6*thermal_power_plants['estimated_emission_factor_mol_kwh']*thermal_power_plants['estimated_generation_gwh_'+m]

    
#  Plot the no2 emitted quantity per power plant in mol
fig = plt.figure(figsize=(10,10))
color_perfuel = {'Oil':'b','Coal':'k','Gas':'g'}
for i in range(len(thermal_power_plants)):
    #fuel.append(thermal_power_plants.loc[i,'estimated_generation_gwh_Jan'])
    emission_per_month = np.zeros(12)
    for im in range(12):
        emission_per_month[im] = thermal_power_plants.loc[i,'estimated_emission_mol_'+months_list[im]]
    fuel_name = thermal_power_plants.loc[i,'primary_fuel']   
    plt.plot(months_list,emission_per_month,color=color_perfuel[fuel_name],label=fuel_name)
plt.legend()
plt.xlabel('month')   
plt.ylabel('No2 emission (mol)')  
image = main_folder+'eie_data/s5p_no2/s5p_no2_20180708T172237_20180714T190743.tif'
image_band = rio.open(image)
image_band.descriptions
s5p_filenames =  np.sort(os.listdir(folder_s5p_no2))
images_s5p = {b:np.array([rio.open(folder_s5p_no2+file).read(b)[::-1,:] for file in s5p_filenames]) for b in range(1,5) }

start_date = datetime.date(2018, 7, 7)
end_date = start_date+  datetime.timedelta(days=5)
# we apply a gaussian filter and average over the periode start_date -> end_date
im_no2 = nan_filter_mean(folder_s5p_no2,images_s5p,start_date,end_date,band=2,sigma=5.,acceptable_nan=0.1)
fig = plt.figure(figsize=(8,8))
plt.imshow(im_no2,cmap='jet')
plt.colorbar()
NO2_concentration_month = np.zeros(12)
generated_energy = np.zeros(12)
date = []
debut_date_list = [  datetime.date(2018, i, 1) if (i<13) else datetime.date(2019, i-12, 1)   for i in range(7,20)]

for i in range(0,12):
  
    debut_date = debut_date_list[i]
    end_date = debut_date_list[i+1]
   
    
    im_no2 = nan_filter_mean(folder_s5p_no2,images_s5p,debut_date,end_date,band=2,sigma=0.,acceptable_nan=0.1)
    month = debut_date.ctime().split(' ')[1]
    date.append(month)
    generated_energy[i]  = thermal_power_plants['estimated_generation_gwh_'+month].sum()
    NO2_concentration_month[i] = im_no2.sum()
    
NO2_concentration_month_normalize = NO2_concentration_month/NO2_concentration_month.mean()
generated_energy_normalize = generated_energy/generated_energy.mean()

plt.plot(date,NO2_concentration_month_normalize,date,generated_energy_normalize)
plt.legend(['No2 concentration','generated electricity'])
print('the correlation between NO2 concentration and the generated electricity is ',np.corrcoef(NO2_concentration_month,generated_energy)[0,1])

# the mean no2 concentration on one year
# the mean no2 concentration on one year
start_date = datetime.date(2018, 7, 7) 
end_date = start_date+  datetime.timedelta(days=365)
mean_im_no2_year = nan_filter_mean(folder_s5p_no2,images_s5p,debut_date,end_date,band=2,sigma=5.,acceptable_nan=0.1)
# load the power plant location and the generated elecricity on one year

xy_location, elect_year,emission_per_month = power_plant_location(thermal_power_plants,mean_im_no2_year.shape,lat_limits,lon_limits)

elect_year_norm = elect_year/elect_year.mean()

s = mean_im_no2_year.shape
fig = plt.figure( figsize=(12,9) )
plt.imshow(mean_im_no2_year,cmap='jet')  # +0.2*np.mean(im_no2)*im_plants
for [xy,c] in zip(xy_location,elect_year_norm):
    plt.plot(xy[0],xy[1],'o',markersize=5*c,color='k')
    #plt.annotate('o', xy=xy, xycoords='data'
folder_gfs = main_folder+'eie_data/gfs/'
gfs_filenames =  np.sort(os.listdir(folder_gfs))
images_gfs = {b:np.array([np.array(rio.open(folder_gfs+file).read(b)[::-1,:]) for file in gfs_filenames]) for b in range(4,6) }
nb_days = 2. # we average data over nb_days of days
nb_months = 10
sigma = 5. # sigma to be applied for gaussian filter 
fig, ax = plt.subplots(nb_months,2, figsize=(60, 60))

vmin, vmax = np.nanpercentile(mean_im_no2_year, (5,95))
[vmin, vmax ] = [0.8*vmin,1.2*vmax]
#[vmin, vmax ] = [-5.0*1.e-6,5.0e-6]

for i in range(0,nb_months):
    debut_date = datetime.date(2018, 7, 7) + datetime.timedelta(days=i*30)
    end_date = debut_date+  datetime.timedelta(days=nb_days)
    no2f_filtered = nan_filter_mean(folder_s5p_no2,images_s5p,debut_date,end_date,band=2,sigma=sigma)
    ax[i,0].imshow(no2f_filtered,cmap='jet',vmin=vmin,vmax=vmax)
    
   # ax[i,0].plot(xy[0],xy[1],'o',markersize=20*c,markerfacecolor="None",
    #     markeredgecolor='red', markeredgewidth=5)
    
   # im_prec = no2f_filtered
    ax[i,0].set(title=str(debut_date)+' to '+str(end_date))
    for [xy,c] in zip(xy_location,elect_year_norm):
        ax[i,0].plot(xy[0],xy[1],'o',markersize=20*c,markerfacecolor="None",
         markeredgecolor='red', markeredgewidth=5)
    #plt.colorbar()
    # your code here    
    
    
    
    mean_u = nan_filter_mean(folder_gfs,images_gfs,debut_date,end_date,band=4,sigma=0.)
    mean_v = nan_filter_mean(folder_gfs,images_gfs,debut_date,end_date,band=5,sigma=0.)
    ## reduce the number of dimension 
    [h_step, w_step] = [20,40]
    shape0 = mean_u.shape

    mean_u_red = mean_u[::h_step,::w_step]
    mean_v_red = mean_v[::h_step,::w_step]
    shape = mean_u_red.shape
    
    x = np.linspace(lon_limits[0],lon_limits[1],shape0[1])
    y = np.linspace(lat_limits[0],lat_limits[1],shape0[0])
    ax[i,1].quiver(x[::w_step],y[::h_step],mean_u_red,mean_v_red,scale=40) 
    for [xy,c] in zip(xy_location,elect_year_norm):
        col = x[int(xy[0])]
        row = y[int(xy[1])]
        ax[i,1].plot(col, row,'o',markersize=20*c,color='r')
    ax[i,1].set(title=str(debut_date)+' to '+str(end_date))

NO2_concentration_month = np.zeros(12)
estimated_NO2 = np.zeros(12)
date = []

#  transform polor coordinate to (x,y)
d = 1
sigma = 0. # we do not need add the gaussian filter to remove noise, 
          # since we are averaging about 30 images for each periode
for i in range(0,12):
    start_date = debut_date_list[i]
    end_date = debut_date_list[i+1]
    T_m = (end_date-start_date).total_seconds()
    im_no2 = nan_filter_mean(folder_s5p_no2,images_s5p,start_date,end_date,band=2,sigma=sigma,acceptable_nan=0.1)

    
    im_no2 = im_no2
    no2_north_face = np.mean(im_no2[-d:,:],axis=0)
    no2_south_face = np.mean(im_no2[:d,:],axis=0)
    no2_west_face = np.mean(im_no2[:,:d],axis=1)
    no2_east_face = np.mean(im_no2[:,-d:],axis=1)

    
    mean_u = nan_filter_mean(folder_gfs,images_gfs,start_date,end_date,band=4,sigma=sigma)
    mean_v = nan_filter_mean(folder_gfs,images_gfs,start_date,end_date,band=5,sigma=sigma)
    mean_u = mean_u
    mean_v = mean_v
    speed_north_face = np.mean(mean_v[-d:,:],axis=0)
    speed_south_face = np.mean(mean_v[:d,:],axis=0)
    
    speed_east_face = np.mean(mean_u[:,-d:],axis=1)
    speed_west_face = np.mean(mean_u[:,:d],axis=1)
    
    flux_north = dx*(speed_north_face*no2_north_face).sum()
    flux_south = -dx*(speed_south_face*no2_south_face).sum()
    flux_east = dy*(speed_east_face*no2_east_face).sum()
    flux_west = -dy*(speed_west_face*no2_west_face).sum()
 
    month = start_date.ctime().split(' ')[1]
    date.append(month)
    estimated_NO2[i]  = thermal_power_plants['estimated_emission_mol_'+month].sum()-(flux_north+flux_south+flux_east+flux_west)*30*24*3600
    
    NO2_concentration_month[i] = dx*dy*im_no2.sum()
    
     
    
    
NO2_concentration_month_normalize = NO2_concentration_month/NO2_concentration_month.mean()
estimated_NO2_normalize = estimated_NO2/estimated_NO2.mean()

plt.plot(date,NO2_concentration_month_normalize,date,estimated_NO2_normalize)
plt.legend(['No2 concentration','estimated_NO2_normalize'])
print('the correlation between NO2 concentration and the generated electricity is ',np.corrcoef(NO2_concentration_month,estimated_NO2_normalize)[0,1])


# specify the time slot
start_time = datetime.datetime(2018, 9, 23,12,0,0) 
end_time = start_time+  datetime.timedelta(days=5)



time_s5p_list = get_time_stamp_list_s5p(s5p_filenames)
time_gfs_list = get_time_stamp_list_gfs(gfs_filenames)


timestamp_no2,im_no2 = get_data_in_time_slot(images_s5p[2],time_s5p_list,start_time,end_time)



timestamp_no2,im_no2 = remove_bad_images(timestamp_no2,im_no2,nan_limit_ratio=0.2)

# We fix start_time as the time origine 

t_no2 = np.array([(t- start_time).total_seconds() for t in timestamp_no2])

t_gfs = np.array([(t- start_time).total_seconds() for t in time_gfs_list])



# gfs contains wind speed each 6hours and in 2D map of 'real resolution (4,9)'
# continuous_wind_speed Object allows:- interpolate the (4,9) wind speed map to (148,475)
#                                     - get wind speed-map in continuous time by interpolating over time
wind = continuous_wind_speed(t_gfs,images_gfs[4],images_gfs[5],X,Y,(4,9))

#  ---- downsample the resolution to by factor  reduction_factor
reduction_factor = [4,5]
new_shape = (int(shape0[0]/reduction_factor[0]),int(shape0[1]/reduction_factor[1]))
wind.downsample(new_shape)
im_no2 = np.array(  [rebin(im, new_shape) for im in im_no2])
    
    
X = np.pi/180*Rt*np.cos(lat*np.pi/180)*np.linspace(lon_limits[0],lon_limits[1],new_shape[1])
Y = np.pi/180*Rt*np.linspace(lat_limits[0],lat_limits[1],new_shape[0])
dx = np.abs(X[1]-X[0])
dy = np.abs(Y[1]-Y[0])
xy_loc, elec_year,emission_monthly = power_plant_location(thermal_power_plants,mean_im_no2_year.shape,lat_limits,lon_limits)

XYp = [[x/reduction_factor[1],y/reduction_factor[0]]   for [x,y] in xy_loc]
Em_p = [em['Nov']/(dx*dy)  for em in emission_monthly]
m = 3
fig,ax=  plt.subplots(m-1,3,figsize=(60,40))
for s in range(1,m):
    # initilize the no2 solver
    no2 = no2_diffusion_convection_solver(im_no2[s-1],X,Y,t_no2[s-1],t_no2[s],XYp, Em_p,D=0.6)
    
    bound = no2.boundary_condition(im_no2[s-1])
    
    no2.simulate_diffusion_convection_source_terms(wind,Cs=0.8)
    
    
    #----------------- Plot the solution-------------#
    im = ax[s-1,0].imshow(no2.C0[:,:],cmap='jet')
    ax[s-1,0].set(title='initial concentration at t= '+str(timestamp_no2[s-1]))

    cbar = fig.colorbar(im,ax=ax[s-1,0], format='%1.2e')
    cbar.ax.tick_params(labelsize=25)
    im =  ax[s-1,1].imshow(no2.C[:,:],cmap='jet')
    ax[s-1,1].set(title='simulated  no2 concentration at t= '+str(timestamp_no2[s]))
    cbar = fig.colorbar(im,ax=ax[s-1,1], format='%1.2e')
    cbar.ax.tick_params(labelsize=25)
    im = ax[s-1,2].imshow(im_no2[s],cmap='jet')
    ax[s-1,2].set(title=' no2 concentration at t= '+str(timestamp_no2[s]))
    cbar = fig.colorbar(im ,ax=ax[s-1,2], format='%1.2e')
    cbar.ax.tick_params(labelsize=25)
    #---------------------------------------------------#
# we keep only power plant with important capacity,to avoid a kind of an overfitting of low power plant capacities that are close 
# to other sources of No2
powerplant_filtered_capacity = thermal_power_plants[thermal_power_plants.capacity_mw>50.].reset_index()
xy_p, el_p,em_p = power_plant_location(powerplant_filtered_capacity,shape0,lat_limits,lon_limits)
name_p = np.array(powerplant_filtered_capacity['name'].tolist())
xy_p = np.array(xy_p)
el_p = np.array(el_p)
em_p = np.array(em_p)
# In case of more than one power plants are close to each ||xy_loc[i]-xy_loc[j]||_2<xy_dist_threshold, we take the barycenter of 
# these close power plant weigthed by their respective capacity

xy_dist_threshold = 4
XY_reduced = []
Ep_reduced = []
name_reduce = []
i = 0
while(len(xy_p)>i):
    delta_x = np.array(xy_p[i,:])-xy_p
    I =  np.sum(delta_x**2,axis=1)<xy_dist_threshold
    XY_reduced.append(np.mean(el_p[I]*xy_p[I,:],axis=0)/np.sum(el_p[I]))
                      
    Ep_reduced.append({m:  np.sum(np.array([e[m] for e in em_p[I]])) for m in months_list})
    name_reduce.append(name_p[I])
    I[i] = False  # we keep 
    xy_p = xy_p[~I]
    em_p = em_p[~I]
    el_p = el_p[~I]
    name_p = name_p[~I]
    i = i+1
print('The final number of source positions (power plant positions or barycenter of some power plant positions) that we are going to use: ',len(XY_reduced))
c_basis = []

xy_p = [[x/reduction_factor[1],y/reduction_factor[0]]   for [x,y] in XY_reduced]  # reduce xy-index PP positions by the same factor as for no2 2d-map


no2 = no2_diffusion_convection_solver(im_no2[0],X,Y,t_no2[0],t_no2[1],[], [],D=0.6)

bound = no2.boundary_condition(im_no2[0])

no2.simulate_diffusion_convection_source_terms(wind,Cs=0.9)

c_bar = no2.C

for xyp in xy_p:
    c0 = np.zeros(im_no2[0].shape)
    no2 = no2_diffusion_convection_solver(c0,X,Y,t_no2[0],t_no2[1],[xyp], [1.],D=0.6)

    bound = no2.boundary_condition(c0)

    no2.simulate_diffusion_convection_source_terms(wind,Cs=0.9)
    c_basis.append(no2.C)
fig = plt.figure()

plt.imshow(c_bar,cmap='jet')
plt.title('$\overline{c}$')
plt.colorbar(format='%1.2e')
#ax.set(title='$\overline{c}$')
fig,ax=  plt.subplots(len(c_basis)//2,2,figsize=(20,16))
for i in range(len(c_basis)):
    ix = i%2
    iy = i//2
    im = ax[iy,ix].imshow(c_basis[i],cmap='jet')
    ax[iy,ix].set(title='basis element: $c_'+str(i)+'$')
    fig.colorbar(im, ax=ax[iy,ix],format='%1.2e')
sh = c_basis[0].shape
Xint,Yint = np.meshgrid(np.linspace(0,sh[1]-1,sh[1]),np.linspace(0,sh[0]-1,sh[0]))
dr = np.array([np.sqrt((Xint-xy[0])**2+ (Yint-xy[1])**2) for xy in XYp  ])
roi = np.min(dr,axis=0)<7.
plt.imshow(roi)
plt.title('Region of Interest')
coef,no2_fitted = emission_factor_lsq_with_lambda_nnls(c_bar,c_basis,im_no2[1],roi,lambda_lim=0.5)

fig,ax=  plt.subplots(1,2,figsize=(40,14))
im = ax[0].imshow(no2_fitted,cmap='jet',vmin=0,vmax=2.5e-5)
ax[0].set(title='fitted solution')
cbar =fig.colorbar(im,ax=ax[0],format='%1.2e')
cbar.ax.tick_params(labelsize=25)

im = ax[1].imshow(im_no2[1],cmap='jet',vmin=0,vmax=2.5e-5)
ax[1].set(title='ground true soltution')
cbar = fig.colorbar(im,ax=ax[1],format='%1.2e')
cbar.ax.tick_params(labelsize=25)
print('emission rate coefficient',coef)


#  delta between the fitted solution and the reference solution

fig,ax = plt.subplots(1,2,figsize=(20,10))
im = ax[0].imshow(np.abs(no2_fitted-im_no2[1])*roi,cmap='jet',vmax=3.e-5)
ax[0].set(title='delta between the fitted solution and groundtruth on region of interest')
fig.colorbar(im,ax=ax[0],format='%1.2e')
im = ax[1].imshow(np.abs(no2_fitted-im_no2[1]),cmap='jet',vmax=3.e-5)
ax[1].set(title='delta between the fitted solution and groundtruth')
fig.colorbar(im,ax=ax[1],format='%1.2e')

delta = np.abs(no2_fitted-im_no2[1])
print(' relative std error in ROI ',np.std(delta[roi])/mean_im_no2_year.mean())
print(' relative std error in all the domain ',np.std(delta)/mean_im_no2_year.mean())
columns = ['date','std_erro','std_erro_roi'] + ['emission_rate_'+str(i)  for i in range(8)]

print(columns)
pd_sim = pd.DataFrame(columns=columns)

start_time = datetime.datetime(2018, 7, 8,12,0,0)  
end_time = start_time+  datetime.timedelta(days=365)



time_s5p_list = get_time_stamp_list_s5p(s5p_filenames)
time_gfs_list = get_time_stamp_list_gfs(gfs_filenames)


timestamp_no2,im_no2 = get_data_in_time_slot(images_s5p[2],time_s5p_list,start_time,end_time)



timestamp_no2,im_no2 = remove_bad_images(timestamp_no2,im_no2,nan_limit_ratio=0.2)

# We fix start_time as the time origine 

t_no2 = np.array([(t- start_time).total_seconds() for t in timestamp_no2])

t_gfs = np.array([(t- start_time).total_seconds() for t in time_gfs_list])



# gfs contains wind speed each 6hours and in 2D map of 'real resolution (4,9)'
# continuous_wind_speed Object allows:- interpolate the (4,9) wind speed map to (148,475)
#                                     - get wind speed-map in continuous time by interpolating over time
wind = continuous_wind_speed(t_gfs,images_gfs[4],images_gfs[5],X,Y,(4,9))

#  ---- downsample the resolution by factor of reduction_factor to speedup the simulation
reduction_factor = [4,5]
new_shape = (int(shape0[0]/reduction_factor[0]),int(shape0[1]/reduction_factor[1]))
wind.downsample(new_shape)
im_no2 = np.array(  [rebin(im, new_shape) for im in im_no2])


X = np.pi/180*Rt*np.cos(lat*np.pi/180)*np.linspace(lon_limits[0],lon_limits[1],new_shape[1])
Y = np.pi/180*Rt*np.linspace(lat_limits[0],lat_limits[1],new_shape[0])



xy_p = [[x/reduction_factor[1],y/reduction_factor[0]]   for [x,y] in XY_reduced]  # reduce xy-index PP positions by the same factor as for no2 2d-map


im_no2[0][im_no2[0]<=1.e-8] = 1.e-8  # remove negative values
for s in range(1,len(im_no2)):

    if((t_no2[s]-t_no2[s-1])<3600*24*1.5):
        no2 = no2_diffusion_convection_solver(im_no2[s-1],X,Y,t_no2[s-1],t_no2[s],[], [],D=0.6)

        bound = no2.boundary_condition(im_no2[s-1])

        no2.simulate_diffusion_convection_source_terms(wind,Cs=0.9)

        c_bar = no2.C
        c_basis = []
        for xyp in xy_p:
            c0 = np.zeros(im_no2[s-1].shape)
            no2 = no2_diffusion_convection_solver(c0,X,Y,t_no2[s-1],t_no2[s],[xyp], [1.],D=0.6)

            bound = no2.boundary_condition(c0)

            no2.simulate_diffusion_convection_source_terms(wind,Cs=0.9)
            c_basis.append(no2.C)
        im_no2[s][im_no2[s]<=1.e-8] = 1.e-8   # remove negative values
        emission_coeff,no2_fitted = emission_factor_lsq_with_lambda_nnls(c_bar,c_basis,im_no2[s],roi,lambda_lim=0.5)

        delta = np.abs(no2_fitted-im_no2[s])
        row_pd = {'emission_rate_'+str(i):emission_coeff[i]*(no2.dx*no2.dy)/(t_no2[s]-t_no2[s-1]) for i in range(8)}
        row_pd['date'] = timestamp_no2[s-1]

        row_pd['std_erro'] = np.std(delta)/np.mean(im_no2[s])
        row_pd['std_erro_roi'] = np.std(delta[roi])/np.mean(im_no2[s][roi])

        pd_sim = pd_sim.append(row_pd, ignore_index=True)
        print(' ')
        print(' relative std error in ROI ',np.std(delta[roi])/mean_im_no2_year.mean())
        print(' relative std error in all the domain ',np.std(delta)/mean_im_no2_year.mean())



    
fig, ax = plt.subplots(figsize=(10,8))
pd_sim_filtered = pd_sim[pd_sim.std_erro_roi<0.5]

pd_sim_filtered.plot(x='date',y=['emission_rate_'+str(i) for i in range(8)], ax=ax)
ax.legend(['emission rate of '+str(name_reduce[i])+'power plant' for i in range(8)])
ax.set(ylabel='mol/s')
pd_sim_filtered['month'] = pd_sim_filtered.date.apply(lambda r:r.month)
pd_sim_monthly = pd_sim_filtered.groupby('month').mean().reset_index()
pd_sim_monthly
fig, ax = plt.subplots(figsize=(10,8))

pd_sim_monthly.plot(x='month',y=['emission_rate_'+str(i) for i in range(8)], ax=ax,marker='.')

ax.legend(['emission rate of '+str(name_reduce[i])+' power plant' for i in range(8)])
ax.set(ylabel='mol/s')