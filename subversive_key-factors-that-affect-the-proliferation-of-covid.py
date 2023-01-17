import pandas as pd 

import numpy as np

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans



import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/foo.csv')

df.head(5)
y = df['country']



cols = ['population_density', 'population_living_in_urban_areas',

       'proportion_of_population_with_basic_handwashing_facilities_on_premises',

       'people_using_at_least_basic_sanitation_services',

       'mortality_rate_under_5', 'prevalence_of_undernourishment',

       'physicians_density', 'current_health_expenditure_per_capita',

       'maternal_mortality_ratio']





x = df[cols]

x.head(5)
scaler = StandardScaler().fit(x)

rescaled_x = scaler.transform(x)

rescaled_x = pd.DataFrame(rescaled_x)

rescaled_x.columns = cols

rescaled_x.head(5)
error = []

for _ in range(1,21):

    kmeans = KMeans(n_clusters=_).fit(rescaled_x)

    kmeans.fit(rescaled_x)

    error.append(kmeans.inertia_)

plt.style.use('dark_background')

plt.plot(range(1,21), error)

plt.title('Elbow Method on Rescaled Data')

plt.xlabel('Number of Clusters')

plt.ylabel('Error')

plt.show()
kmeans20 = KMeans(n_clusters=20)

y_kmeans20 = kmeans.predict(rescaled_x)

y_kmeans20 = pd.DataFrame({'cluster1': y_kmeans20})



df_clust1 = pd.concat([rescaled_x, y, y_kmeans20], axis=1)
df_clust13 = df_clust1.query('cluster1 == 14')

df_clust13.head(14)
clust13_country = list(df_clust13['country'])

clust13_country



df = pd.read_csv('../input/foo.csv')



df_clust13 = df.loc[df['country'].isin(clust13_country)]



cols = ['country',

       'population_density', 'population_living_in_urban_areas',

       'proportion_of_population_with_basic_handwashing_facilities_on_premises',

       'people_using_at_least_basic_sanitation_services',

       'mortality_rate_under_5',

       'prevalence_of_undernourishment', 

       'physicians_density',

       'current_health_expenditure_per_capita', 'maternal_mortality_ratio']



df_clust13 = df_clust13[cols]

df_clust13
cols = ['population_density', 'population_living_in_urban_areas',

       'proportion_of_population_with_basic_handwashing_facilities_on_premises',

       'people_using_at_least_basic_sanitation_services',

       'mortality_rate_under_5', 'prevalence_of_undernourishment',

       'physicians_density', 'current_health_expenditure_per_capita',

       'maternal_mortality_ratio']



for _ in cols:

    sns.set_style('darkgrid')

    fig_dims = (8, 5)

    fig, ax = plt.subplots(figsize=fig_dims)

    

    foo = df_clust13.sort_values(_, ascending=False)

    sns.barplot(x=foo[_], y=foo.country, palette='Blues_d')

    plt.ylabel('Country')

    plt.show()