import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os # directory stuff

import matplotlib.pyplot as plt # plotting data

%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D # 3D MPL

import seaborn as sns # a library built on top of matplotlib, makes statistical plotting easier and prettier

from sklearn.cluster import KMeans # clustering

import math



#print(os.listdir("../input"))
raw_data = pd.read_csv('../input/cumulative.csv')
print('Here are all the columns:\n')

print(raw_data.columns.tolist())
# aforementioned columns we don't want

koi_pond = ['koi_pdisposition', 'koi_tce_plnt_num', 'koi_tce_delivname', 'kepler_name']



# gonna iterate through the columns of the raw data to add the uncertainty/error ones to the drop-list (koi_pond)

cols = raw_data.columns



# to make dropping the error columns easier, I'm just going to go through the whole list and drop every column with 'err' in the name. 

for c in cols:

    if 'err' in c:

        koi_pond.append(c)



# finalize changes, but hold on to the raw data in case we need it later

data = raw_data.drop(koi_pond, axis = 1)



print("Dropped:\n\n", koi_pond)

print(f"\nYour dataset had {raw_data.shape[1]} columns.\nIt now has {data.shape[1]} columns.")
data = data.dropna()

print(f"The dataset had {raw_data.shape[0]} rows. It now has {data.shape[0]} rows.\n({raw_data.shape[0]-data.shape[0]} rows were dropped, leaving you with {round(((data.shape[0]/raw_data.shape[0])*100),2)}% of the original number of entries.)")
data.rename(columns={'kepid':'kepler_id','kepoi_name':'koi_name','koi_disposition':'plnt_disposition', 'koi_score':'plnt_disp_confidence',

                     'koi_fpflag_nt':'flag_nTransitLk', 'koi_fpflag_ss':'flag_scndEvent', 'koi_fpflag_co':'flag_centroidOffset', 

                     'koi_fpflag_ec':'flag_ephMatch', 'koi_period':'orbital_period','koi_time0bk':'transit_epoch','koi_impact':'impact_parameter',

                     'koi_duration':'transit_duration','koi_depth':'transit_depth', 'koi_prad':'planetary_radius', 'koi_teq':'equ_temp',

                     'koi_insol':'insolation_flux','koi_model_snr':'transit_sigToNoise','koi_steff':'stellar_eff_temp','koi_slogg':'stellar_surf_gravity',

                     'koi_srad':'stellar_radius','ra':'right_acension','dec':'declination','koi_kepmag':'kepler_magnitude'

                    }, inplace = True)



data.tail()
columns_list = data.columns.tolist()

print(columns_list)
desc_list = [

    'the row ID',

    'target identification number as listed in the Kepler Input Catalog',

    'number used to track a KOI (Kepler Object of Interest)',

    'exoplanet status of this KOI from the Exoplanet Archive. Values are either FALSE POSITIVE, CANDIDATE, or CONFIRMED [exoplanet]',

    'value between 0 and 1 indicating confidence in KOI disposition.  For CANDIDATEs, a higher value indicates more confidence in its disposition, while for FALSE POSITIVEs, a higher value indicates less confidence in that disposition.',

    'not transit-like. flag indicating a KOI whose light curve is not consistent with that of a transiting planet. This includes, but is not limited to, instrumental artifacts, non-eclipsing variable stars, and spurious (very low SNR) detections.',

    'stellar eclipse. flag indicating a KOI that is observed to have a significant secondary event, transit shape, or out-of-eclipse variability, which indicates that the transit-like event is most likely caused by an eclipsing binary. ',

    'centroid offset. flag indicating a KOI where the source of the signal is from a nearby star, as inferred by measuring the centroid location of the image both in and out of transit, or by the strength of the transit signal in the target\'s outer (halo) pixels as compared to the transit signal from the pixels in the optimal (or core) aperture',

    'ephemeris match indicates contamination. flag indicating a KOI that shares the same period and epoch as another object and is judged to be the result of flux contamination in the aperture or electronic crosstalk.',

    'orbital period (days)',

    'The time corresponding to the center of the first detected transit in Barycentric Julian Day (BJD) minus a constant offset of 2,454,833.0 days. The offset corresponds to 12:00 on Jan 1, 2009 UTC.',

    'The sky-projected distance between the center of the stellar disc and the center of the planet disc at conjunction, normalized by the stellar radius.',

    'Transit Duration (hours). The duration of the observed transits. Duration is measured from first contact between the planet and star until last contact.',

    'Transit Depth (parts per million). The fraction of stellar flux lost at the minimum of the planetary transit.',

    'Planetary radius (Earth radii). The product of the planet-star radius ratio and the stellar radius.',

    'Equilibrium Temperature (Kelvin). An approximate temperature of the planet/object.',

    'Insolation Flux [Earth flux]. Insolation flux is another way to give the equilibrium temperature. It depends on the stellar parameters (specifically the stellar radius and temperature), and on the semi-major axis of the planet. It\'s given in units relative to those measured for the Earth from the Sun.',

    'Transit Signal-to-Noise. Transit depth normalized by the mean uncertainty in the flux during the transits.',

    'Stellar Effective Temperature (Kelvin). The photospheric temperature of the star the KOI orbits.',

    'Stellar Surface Gravity (log_10 (cm/s^2)). The base-10 logarithm of the acceleration due to gravity at the surface of the star.',

    'Stellar Radius (solar radii). The photospheric radius of the star.',

    'KIC Right Acension',

    'KIC Declination',

    'Kepler-band magnitude',

]
columns_desc = dict(zip(columns_list, desc_list))

columns_desc['flag_ephMatch']
# function that returns a classification for a star



star_types = ['O','B','A','F','G','K','M']



def spectral_classify(temp):

    if temp > 33000:

        return star_types[0]

    elif temp > 10000:

        return star_types[1]

    elif temp > 7500:

        return star_types[2]

    elif temp > 6000:

        return star_types[3]

    elif temp > 5200:

        return star_types[4]

    elif temp > 3700:

        return star_types[5]

    else:

        return star_types[6]
spectral_classes = [] # will hold the results of spectral classification



for temp in data['stellar_eff_temp'].values: # classify all the stars in the dataset

    spectral_classes.append(spectral_classify(temp))



data['stellar_spectral_cl'] = spectral_classes

print('Spectral classes added to dataset.')
# OBAFGKM palette: red, orange, yellow, yellow-white, white, blue-white, blue

star_colors = ['#fc0303','#fc8403','#fcf403', '#fffb91', '#ffffff','#94b6ff','#0877ff']

star_palette = dict(O=star_colors[6], B=star_colors[5], A=star_colors[4], F=star_colors[3], 

                             G=star_colors[2],K=star_colors[1],M=star_colors[0])
plt.style.use("dark_background")

ax = sns.catplot(x="stellar_spectral_cl",col='plnt_disposition', kind="count", order=['O','B','A','F','G','K','M'], palette=dict(O=star_colors[6], B=star_colors[5], A=star_colors[4], F=star_colors[3], G=star_colors[2],K=star_colors[1],M=star_colors[0]), data=data)

ax.set(xlabel='spectral class')

ax.despine(left=True, bottom=False)



plt.show()
#sns.set_style("dark")

plt.style.use("dark_background")

sns.set_context('talk')

plt.figure(figsize=(17,10))

ax = sns.scatterplot(x='stellar_radius',y='stellar_surf_gravity', data=data, 

                hue='stellar_spectral_cl', hue_order=['O','B','A','F','G','K','M'],

                palette=star_palette, 

                markers = '.',

                alpha=0.4,

                linewidth=0,

                size = 'stellar_radius'

                    )



ax.set(xlabel='radius (solar radii)', ylabel='surface gravity log(cm/(s*s))')



plt.title('Star Surface Gravity vs. Star Radius')

plt.show()
# try different values of k

max_n_clusters = 10 # max number of k to try

iterations = 5000 # how many times to try each k



category_list = ['stellar_radius','stellar_surf_gravity','stellar_eff_temp']



squared_distance = np.zeros(max_n_clusters)

for k in range(2,max_n_clusters):

    kmeans = KMeans(n_clusters = k, max_iter = iterations).fit(data[category_list])

    squared_distance[k] = kmeans.inertia_

    print(k,sep=' ', end=' ', flush=True)

    



plt.figure(figsize=(5,5))

plt.plot(squared_distance,c='r')

plt.xlim((2,max_n_clusters))

plt.xlabel('number of clusters (k)')

plt.ylabel('avg dist to centers')

plt.title('Elbow Plot for K-Means Clustering')

plt.show()
numClusters = 4 # set number of clusters



# create and fit model

km_ = KMeans(n_clusters = numClusters, max_iter = iterations)

km_.fit(data[category_list])

km_.labels_ # cluster assignments
data['star_cluster'] = km_.labels_
data.groupby('star_cluster')[category_list].mean()
fig = plt.figure(figsize=(20,10))

sns.set_context('paper')

data['star_cluster'] = data['star_cluster'].astype(str)



km_palette1 = ['r','y','g','b',]



plt.subplot(121)



if numClusters == len(km_palette1):



    kmp1 = sns.scatterplot(x='stellar_surf_gravity',y='stellar_eff_temp',data=data,

                        linewidth=0, alpha=0.6,

                        size='stellar_radius',

                        hue='star_cluster', palette=km_palette1)

    

    kmp1.set_xlabel('surface gravity')

    kmp1.set_ylabel('temperature')

else:

    print("Palette is wrong length.")

    



plt.title('Gravity vs. Temperature, with Radius as marker size')





# SECOND GRAPH



# TODO: Convert cluster data to numeric



star_cluster_colors = []



for val in data['star_cluster']:

    if float(val) != 0:

        star_cluster_colors.append(1/float(val))

    else:

        star_cluster_colors.append(0)



kmp2 = fig.add_subplot(122, projection='3d')

kmp2.scatter(data['stellar_surf_gravity'], data['stellar_radius'], data['stellar_eff_temp'], c=star_cluster_colors, marker='.', s=1, alpha=1, cmap='RdBu')



kmp2.set_xlabel('surface gravity')

kmp2.set_ylabel('radius')

kmp2.set_zlabel('temperature')

plt.title('Gravity vs. Temperature vs. Radius')



plt.show()
# calculate the correlation matrix

corr = data.corr()



plt.figure(figsize = (12,8))

# plot the heatmap

ax = sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns, annot=False)

plt.show()