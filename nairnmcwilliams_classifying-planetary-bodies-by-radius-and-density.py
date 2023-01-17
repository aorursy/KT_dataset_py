import pandas as pd

import matplotlib.pyplot as plt
planetary_data = pd.read_csv('../input/solar-system-radius-and-density-data/planetary_data.csv', header=0)

planetary_data = planetary_data.set_index('Body', drop=True)



radius = planetary_data['RadiusSI']

density = planetary_data['DensitySI']



plt.figure(figsize=(12, 9))

plt.scatter(density, radius)

plt.title('Variation in planetary radii with density')

plt.xlabel('Density $kg\,m^{-3}$')

plt.ylabel('Radius $km$')

plt.yscale('log')

plt.grid(axis='both', which='both')

plt.show()

plt.close()
import sklearn.cluster as cluster

import numpy as np
colour_dict = {0: 'r', 1: 'g', 2: 'b', 3: 'y'}
# Create the learning algorithm, fitted to the planetary data

log_data = np.array([np.log10(x) for x in [density, radius]]).transpose()

kmeans = cluster.KMeans(n_clusters=len(colour_dict), random_state=0).fit(log_data)



# Determine the clusters using this algorithm and fit this to the planetary data

planetary_data['Cluster'] = kmeans.predict(log_data)
category_labels = {0: 'Terrestrial planet', 1: 'Asteroid/comet', 2: 'Icy moon', 3: 'Giant planet'}

num_category_labels = len(category_labels)



mercury_cluster = planetary_data.loc['Mercury', 'Cluster']

phobos_cluster = planetary_data.loc['Phobos', 'Cluster']

mimas_cluster = planetary_data.loc['Mimas', 'Cluster']

jupiter_cluster = planetary_data.loc['Jupiter', 'Cluster']



# Assign temporary values to the cluster of each body. This prevents a body from another category being assigned the same value.

planetary_data.loc[planetary_data['Cluster'] == mercury_cluster, 'Cluster'] = mercury_cluster + num_category_labels

planetary_data.loc[planetary_data['Cluster'] == phobos_cluster, 'Cluster'] = phobos_cluster + num_category_labels

planetary_data.loc[planetary_data['Cluster'] == mimas_cluster, 'Cluster'] = mimas_cluster + num_category_labels

planetary_data.loc[planetary_data['Cluster'] == jupiter_cluster, 'Cluster'] = jupiter_cluster + num_category_labels



# Reassign the cluster values.

planetary_data['Cluster'] = planetary_data['Cluster'] - num_category_labels
def set_categories(data, category_labels):

    data['Category'] = [category_labels[x] for x in data['Cluster']]



set_categories(planetary_data, category_labels)

planetary_data
from matplotlib.backends.backend_pdf import PdfPages



def plot_planetary_data(planetary_data, colour_dict, category_labels, density_column, radius_column, title, output_pdf=None):

    if output_pdf is not None:

        pp = PdfPages(output_pdf)

        plt.figure(figsize=(12, 9))

    else:

        pp = None

    

    plt.title(title)

    plt.xlabel('Density ($kg\,m^{-3}$)')

    plt.ylabel('Radius ($m$)')

    plt.grid(axis='both', which='both')

    plt.yscale('log')

    

    for i in range(0, len(colour_dict)):

        subdata = planetary_data.loc[planetary_data['Cluster'] == i]

        plt.scatter(subdata[density_column], subdata[radius_column], color=colour_dict[i], label=category_labels[i])

    

    plt.legend(loc='best')



    if pp is not None:

        pp.savefig()

    plt.show()

    plt.close()

    

    if pp is not None:

        pp.close()
plot_planetary_data(

    planetary_data,

    colour_dict,

    category_labels,

    'DensitySI',

    'RadiusSI',

    'K-means categorisation of planetary data'

)
category_labels2 = {0: 'Terrestrial planet', 1: 'Asteroid/comet', 2: 'Icy moon', 3: 'Giant planet'}



planetary_data2 = planetary_data.copy()

planetary_data2['Cluster'] = None



# Classify each body using manual rules

planetary_data2.loc[planetary_data2['RadiusSI'] > 1E4, 'Cluster'] = 3

planetary_data2.loc[

    (planetary_data2['RadiusSI'] > 1E3) &

    (planetary_data2['DensitySI'] > 2.5E3) &

    (planetary_data2['Cluster'].isnull()), 'Cluster'] = 0

planetary_data2.loc[planetary_data2['RadiusSI'] < 300, 'Cluster'] = 1

planetary_data2.loc[planetary_data2['Cluster'].isnull(), 'Cluster'] = 2



set_categories(planetary_data2, category_labels2)

planetary_data2
plot_planetary_data(

    planetary_data2,

    colour_dict,

    category_labels2,

    'DensitySI',

    'RadiusSI',

    'Manual categorisation of planetary data'

)
import sklearn.svm as svm



# Create the learning algorithm, fitted to the planetary data

log_data = np.array([np.log10(x) for x in [density, radius]]).transpose()

knn = svm.SVC(gamma='scale').fit(log_data, planetary_data2['Cluster'])



planetary_data_svc = planetary_data2.copy()

planetary_data_svc['Cluster'] = knn.predict(log_data)



set_categories(planetary_data_svc, category_labels2)

planetary_data_svc
plot_planetary_data(

    planetary_data_svc,

    colour_dict,

    category_labels,

    'DensitySI',

    'RadiusSI',

    'K-NN categorisation of planetary data'

)
import astropy.units as u

import astropy.units.astrophys as ua
# Read the data

oec = pd.read_csv('../input/open-exoplanet-catalogue/oec.csv', usecols=['PlanetIdentifier', 'PlanetaryMassJpt', 'RadiusJpt'])

oec = oec.set_index('PlanetIdentifier', drop=True)



# Clean the data by removing records that do not specify both mass and radius

oec = oec.loc[(oec['PlanetaryMassJpt'].notnull()) & (oec['RadiusJpt'].notnull())].reindex()



# Describe the radius, mass and volume of each body in SI units

oec['RadiusSI'] = [(x*ua.jupiterRad).to(u.m).value for x in oec['RadiusJpt']]

oec['RadiusKm'] = oec['RadiusSI'] * 1E-3

oec['MassSI'] = [(x*ua.jupiterMass).to(u.kg).value for x in oec['PlanetaryMassJpt']]

oec['VolumeSI'] = 4/3 * np.pi * oec['RadiusSI'] ** 3



# Calculate the density of each body

oec['DensitySI'] = oec['MassSI'] / oec['VolumeSI']
earth_density = planetary_data.loc['Earth', 'DensitySI']

saturn_density = planetary_data.loc['Saturn', 'DensitySI']



oec = oec.loc[

    (oec['DensitySI'] <= 2 * earth_density) &

    (oec['DensitySI'] >= saturn_density / 2)

]
oec_log_data = np.array([np.log10(oec['DensitySI']), np.log10(oec['RadiusKm'])]).transpose()

oec['Cluster'] = knn.predict(oec_log_data)



set_categories(oec, category_labels2)

oec.head()
plot_planetary_data(

    oec,

    colour_dict,

    category_labels2,

    'DensitySI',

    'RadiusKm',

    'K-NN categorisation of exoplanet data'

)