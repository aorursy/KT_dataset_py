# data
import pandas as pd

# visualizations
import seaborn as sns
from matplotlib import pyplot as plt
import pycountry
import plotly.express as px

# preprocessing
import numpy as np
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,FunctionTransformer

# clusters models
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn import metrics
from sklearn.metrics import silhouette_score, silhouette_samples
#from sklearn.metrics.pairwise import cosine_similarity
#from scipy.cluster.hierarchy import dendrogram, linkage
info = pd.read_csv('../input/unsupervised-learning-on-country-data/data-dictionary.csv')
for i, row in info.iterrows():
  print(row['Column Name'],' ---> ', row.Description)
CountryData = pd.read_csv('../input/unsupervised-learning-on-country-data/Country-data.csv')
CountryData.head()
CountryData.isnull().sum()
CountryData.describe()
features = CountryData.columns[1:]

features_group1 = ['child_mort','exports']
features_group2 = list(set(features)-set(features_group1))

g1_transformer = Pipeline(steps=[
    ('log', FunctionTransformer(np.log1p)),
    ('scaler', StandardScaler())
    ])

g2_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('group1', g1_transformer, features_group1),
        ('group2', g2_transformer, features_group2),
        ])
preprocessor.fit(CountryData) 
np_data = preprocessor.transform(CountryData) 
df_data = pd.DataFrame(np_data, columns=features)
#Using Pearson Correlation
plt.figure(figsize=(12,10))
corr_m = CountryData.drop(['country'],axis=1).corr()
sns.heatmap(corr_m, annot=True, cmap=plt.cm.Reds).set_title('Correlation Matrix')
plt.show()
data = df_data.loc[:, features].values


# created a covariance matrix on the standardized data. 
matrix_cov = np.cov(data.T)

# eigendecomposition on covariance matrix
eig_vals, eig_vecs = np.linalg.eig(matrix_cov)
for i in range(len(eig_vals)):
    print(eig_vals[i], eig_vals[i]/np.sum(eig_vals))
#Explained variance
pca = PCA().fit(data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(data)
principalDf = pd.DataFrame(data = principalComponents)

def ISO_Code(df):
    # Countries that are not in the ISO Code
    df['RightCountry'] = [item.replace('Cape Verde', 'Cabo Verde')
                              .replace('Congo, Dem. Rep.', 'Congo, The Democratic Republic of the')
                              .replace('Congo, Rep.', 'Republic of the Congo')
                              .replace('Macedonia, FYR', 'North Macedonia')
                              .replace('Micronesia, Fed. Sts.', 'Micronesia, Federated States of')
                              .replace('South Korea', 'Korea, Republic of')
                              .replace('St. Vincent and the Grenadines', 'Saint Vincent and the Grenadines') for item in df.country]

    list_countries = df['RightCountry'].unique().tolist()

    d_country_code = {}  # To hold the country names and their ISO
    for country in list_countries:
        try:
            country_data = pycountry.countries.search_fuzzy(country)
            country_code = country_data[0].alpha_3
            d_country_code.update({country: country_code})
        except:
            print('could not add ISO 3 code for ->', country)
            # If could not find country, make ISO code ' '
            d_country_code.update({country: ' '})

    for k, v in d_country_code.items():
        df.loc[(df.RightCountry == k), 'iso_alpha'] = v 

    df.loc[df.country.tolist().index('Niger'),'iso_alpha']='NER'
    return df

def get_map(df):
    df = ISO_Code(df)
    fig = px.choropleth(data_frame = df,
                        locations= "iso_alpha",
                        color= "cluster",  # value in column 'Confirmed' determines color
                        hover_name= "country",
                        color_continuous_scale= 'RdYlGn_r',  #  color scale red, yellow green
                        )

    fig.show()

def get_datavisual(df,np_data):
    # Create PCA for data visualization / Dimensionality reduction to 2D graph
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca_model = pca.fit_transform(np_data)
    data_transform = pd.DataFrame(data = pca_model, columns = ['PCA1', 'PCA2'])
    data_transform['Cluster'] = df.cluster

    plt.figure()
    g = sns.scatterplot(data=data_transform, x='PCA1', y='PCA2', palette=sns.color_palette()[:int(df.cluster.nunique())], hue='Cluster')
    title = plt.title('Countries Clusters with PCA')
def run_model(n_c, np_data):
  # Create and fit model
  kmeans = KMeans(n_clusters=n_c,
                  init='k-means++',
                  max_iter=400, 
                  n_init=80, 
                  random_state=0)
  model = kmeans.fit(np_data)

  df = CountryData.copy()
  df['cluster'] = model.labels_

  return df
np_data = principalDf
Sum_of_squared_distances = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, 
                init='k-means++',
                max_iter=400, 
                n_init=80, 
                random_state=0
                ).fit(np_data)
    Sum_of_squared_distances.append(km.inertia_)

#plt.figure(figsize=(10,10))
plt.plot(range(1, 10), Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
df_km3PCA = run_model(3, principalDf)
df_km3PCA.head()
print(df_km3PCA.cluster.value_counts())
df_km3PCA.cluster.hist()
plt.tight_layout()
get_datavisual(df_km3PCA, np_data)
get_map(df_km3PCA)
df_km3PCA.groupby(['cluster']).mean()
from pylab import *

cols = df_km3PCA.columns[1:-1]
for col in cols:
    y0 = sorted(df_km3PCA[df_km3PCA.cluster==0][col].tolist())
    y1 = sorted(df_km3PCA[df_km3PCA.cluster==1][col].tolist())
    y2 = sorted(df_km3PCA[df_km3PCA.cluster==2][col].tolist())

    plt.plot(range(len(y0)),y0,'.', linewidth=4, color='b')
    plt.plot(range(len(y1)),y1, '.', linewidth=4,color='r')
    plt.plot(range(len(y2)),y2,'.', linewidth=4, color='g')
    plt.title(col)
    plt.show()