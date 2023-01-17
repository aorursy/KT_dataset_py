# import of needed libraries



import numpy as np

import pandas as pd

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt



pd.options.display.max_columns = 300



import warnings

warnings.filterwarnings("ignore")
# let's see what our data set look like



df_solar_co = pd.read_csv("../input/30-years-of-european-solar-generation/EMHIRESPV_TSh_CF_Country_19862015.csv")

df_solar_co.head(2)
df_solar_co.columns
country_dict = {

'AT': 'Austria',

'BE': 'Belgium',

'BG': 'Bulgaria',

'CH': 'Switzerland',

'CY': 'Cyprus',

'CZ': 'Czech Republic',

'DE': 'Germany',

'DK': 'Denmark',

'EE': 'Estonia',

'ES': 'Spain',

'FI': 'Finland',

'FR': 'France',

'EL': 'Greece',

'UK': 'United Kingdom',

'HU': 'Hungary',

'HR': 'Croatia',

'IE': 'Ireland',

'IT': 'Italy',

'LT': 'Lithuania',

'LU': 'Luxembourg',

'LV': 'Latvia',

'NO': 'Norway',

'NL': 'Netherlands',

'PL': 'Poland',

'PT': 'Portugal',

'RO': 'Romania',

'SE': 'Sweden',

'SI': 'Slovenia',

'SK': 'Slovakia'

    }
df_solar_co.shape
df_solar_nu = pd.read_csv("../input/30-years-of-european-solar-generation/EMHIRES_PVGIS_TSh_CF_n2_19862015.csv")

df_solar_nu = df_solar_nu.drop(columns=['time_step'])

df_solar_nu.tail(2)
df_solar_nu.shape
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
df_solar_transposed = df_solar_nu[-24*365:].T

df_solar_transposed.tail(2)
def plot_elbow_scores(df_, cluster_nb):

    km_inertias, km_scores = [], []



    for k in range(2, cluster_nb):

        km = KMeans(n_clusters=k).fit(df_)

        km_inertias.append(km.inertia_)

        km_scores.append(silhouette_score(df_, km.labels_))



    sns.lineplot(range(2, cluster_nb), km_inertias)

    plt.title('elbow graph / inertia depending on k')

    plt.show()



    sns.lineplot(range(2, cluster_nb), km_scores)

    plt.title('scores depending on k')

    plt.show()

    

plot_elbow_scores(df_solar_transposed, 20)
df_solar_transposed = df_solar_co[-24*365*10:].T

plot_elbow_scores(df_solar_transposed, 20)
X = df_solar_transposed



km = KMeans(n_clusters=6).fit(X)

X['label'] = km.labels_

print("Cluster nb / Nb of countries in the cluster", X.label.value_counts())



print("\nCountries grouped by cluster")

for k in range(6):

    print(f'\ncluster nb {k} : ', " ".join([country_dict[c] + f' ({c}),' for c in list(X[X.label == k].index)]))