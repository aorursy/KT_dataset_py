# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
import plotly
# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import os
%matplotlib inline
pwd
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

dataset_indicators= pd.read_csv("/kaggle/input/uncover/UNCOVER/HDE_update/inform-covid-indicators.csv")
#dataset_indicator_test_performed= pd.read_csv("/kaggle/input/uncover/HDE/total-covid-19-tests-performed-by-country.csv")
dataset_covid19_cases = pd.read_csv("/kaggle/input/uncover/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-country.csv")
dataset_covid_test_performed= pd.read_csv("/kaggle/input/uncover/UNCOVER/HDE_update/HDE/total-covid-19-tests-performed-by-country.csv")
dataset_indicators.tail(10)
dataset_covid19_cases.tail(3)
dataset_covid_test_performed.tail(5)
dataset_confirmed_cases_by_country = dataset_covid19_cases[['country_region', 'confirmed', 'deaths', 'recovered', 'active']]
#Grouped the confirmed cases, deatjs and recovered cases by country
dataset_confirmed_cases_by_country=dataset_confirmed_cases_by_country.groupby(['country_region']).sum()
dataset_confirmed_cases_by_country.tail(10)
#resetting the index
dataset_confirmed_cases_by_country=dataset_confirmed_cases_by_country.reset_index()
dataset_confirmed_cases_by_country.tail(10)
data = [dict(type='choropleth',
             locations = dataset_confirmed_cases_by_country['country_region'].astype(str),
             z=dataset_confirmed_cases_by_country['confirmed'].astype(int),
             locationmode='country names')]




fig = dict(data=data, 
           layout_title_text="COVID-19 Confirmed cases")


plotly.offline.iplot(fig)
data = [dict(type='choropleth',
             locations = dataset_covid_test_performed['entity'].astype(str),
             z=dataset_covid_test_performed['total_covid_19_tests'].astype(int),
             locationmode='country names')]

fig = dict(data=data, 
           layout_title_text="COVID-19 test performed")

plotly.offline.iplot(fig)
# cleaning the country names for joining
dataset_confirmed_cases_by_country.loc[dataset_confirmed_cases_by_country.country_region=='US','country_region']='United States of America'
dataset_indicators.loc[dataset_indicators.country=='Viet Nam','country']='Vietnam'
dataset_indicators.loc[dataset_indicators.country=='Russian Federation','country']='Russia'
dataset_indicators.loc[dataset_indicators.country=='Korea Republic of','country']='Korea, South'
dataset_indicators.loc[dataset_indicators.country=='Moldova Republic of','country']='Moldova'
dataset_covid_test_performed.loc[dataset_covid_test_performed.entity=='United States','entity']='United States of America'
#replace No Data with 0
dataset_indicators=dataset_indicators.replace("No data", 0)
#renaming country column name to country_region for joing the two dataframes
dataset_indicators.columns=['country_region', 'iso3', 'inform_risk',
       'inform_p2p_hazard_and_exposure_dimension', 'population_density',
       'population_living_in_urban_areas',
       'proportion_of_population_with_basic_handwashing_facilities_on_premises',
       'people_using_at_least_basic_sanitation_services',
       'inform_vulnerability', 'inform_health_conditions',
       'inform_epidemic_vulnerability', 'mortality_rate_under_5',
       'prevalence_of_undernourishment', 'inform_lack_of_coping_capacity',
       'inform_access_to_healthcare',
       'inform_epidemic_lack_of_coping_capacity', 'physicians_density',
       'current_health_expenditure_per_capita', 'maternal_mortality_ratio']
# outer join between cases in different countries and countries health indicators
data_tmp=pd.merge(dataset_confirmed_cases_by_country,dataset_indicators,  on='country_region', how='outer')
data_tmp=pd.merge(data_tmp,dataset_covid_test_performed,  left_on='country_region', right_on='entity',how='left')
data_tmp.tail(2)
#Cleaning the NaN data and data with x
data_tmp=data_tmp.replace(np.nan, 0)
data_tmp=data_tmp.replace('x', 0)
data_tmp.tail(40)
#createing dataset with numeric values
data_k =data_tmp[['total_covid_19_tests', 'confirmed', 'deaths', 'recovered', 'active',
                  'inform_risk', 'inform_p2p_hazard_and_exposure_dimension',
       'population_density', 'population_living_in_urban_areas',
       'proportion_of_population_with_basic_handwashing_facilities_on_premises',
       'people_using_at_least_basic_sanitation_services',
       'inform_vulnerability', 'inform_health_conditions',
       'inform_epidemic_vulnerability', 'mortality_rate_under_5',
       'prevalence_of_undernourishment', 'inform_lack_of_coping_capacity',
       'inform_access_to_healthcare',                
       'current_health_expenditure_per_capita', 'maternal_mortality_ratio']]
from sklearn.cluster import KMeans
#calculating WCSS which is the sum of squares of the distances of each data point represeting a country
#in all clusters to their respective centroids
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
    y_means = kmeans.fit(data_k)
    wcss.append(y_means.inertia_)
#Plotting WCSS to find the number of clusters
plt.plot(range(1,11), wcss)
plt.xlabel("No. of clusters")
plt.ylabel(" Within Cluster Sum of Squares")
plt.show()
kmeans_covid = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10)
y_kmeans = kmeans.fit_predict(data_k)

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(data_k)


#beginning of  the cluster numbering with 1 instead of 0
y_kmeans1=y_kmeans
y_kmeans1=y_kmeans+1

# New Dataframe called cluster
cluster = pd.DataFrame(y_kmeans1)
# Adding cluster to the Dataset1
data_k['cluster'] = cluster
#Mean of clusters
kmeans_mean_cluster = pd.DataFrame(round(data_k.groupby('cluster').mean(),1))
kmeans_mean_cluster
plt.figure(figsize=(10, 8))
plt.scatter(data_k.iloc[:,0], data_k.iloc[:,1],c=y_kmeans, cmap='rainbow')  # plot points with cluster dependent colors
plt.title('Covid Clustering')
plt.xlabel("Test Perfomed by the country")
plt.ylabel("No. of confirmed cases")
plt.show()
data_risk= pd.DataFrame()
data_risk["country"]=data_tmp["country_region"]
data_risk["Risk_Level"]=y_kmeans1

for group in range(1,6):
    countries=data_risk.loc[data_risk['Risk_Level']==group]
    listofcoutries= list(countries['country'])
    print("Group", group, ":", listofcoutries)
data = [dict(type='choropleth',
             colorscale='reds',
             locations =data_risk['country'].astype(str),
             z= data_risk['Risk_Level'].astype(int),
             locationmode='country names')]

fig = dict(data=data, 
           layout_title_text="Country grouped based on Health care quality, no. of COVID-19 cases and tests performed")

plotly.offline.iplot(fig)
# considered heath indicators and test perfomed to understand the impact on confirmed COVID-19 cases
names = ['total_covid_19_tests',  
       'inform_risk', 'inform_p2p_hazard_and_exposure_dimension',
       'population_density', 'population_living_in_urban_areas',
       'proportion_of_population_with_basic_handwashing_facilities_on_premises',
       'people_using_at_least_basic_sanitation_services',
       'inform_vulnerability', 'inform_health_conditions',
       'inform_epidemic_vulnerability', 'mortality_rate_under_5',
       'prevalence_of_undernourishment', 'inform_lack_of_coping_capacity',
       'inform_access_to_healthcare', 'current_health_expenditure_per_capita',
       'maternal_mortality_ratio', 'cluster']
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
mi = mutual_info_regression(data_k[names], data_k['deaths'] )
mi = pd.Series(mi)
mi.index = names
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))
plt.title(" Factor impacting COVID-19 deaths")
#createing dataset with numeric values
data_health =data_tmp[['inform_risk', 'inform_p2p_hazard_and_exposure_dimension',
       'population_density', 'population_living_in_urban_areas',
       'proportion_of_population_with_basic_handwashing_facilities_on_premises',
       'people_using_at_least_basic_sanitation_services',
       'inform_vulnerability', 'inform_health_conditions',
       'inform_epidemic_vulnerability', 'mortality_rate_under_5',
       'prevalence_of_undernourishment', 'inform_lack_of_coping_capacity',
       'inform_access_to_healthcare',                
       'current_health_expenditure_per_capita', 'maternal_mortality_ratio']]
from sklearn.cluster import KMeans
#calculating WCSS which is the sum of squares of the distances of each data point represeting a country
#in all clusters to their respective centroids
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
    y_means = kmeans.fit(data_health)
    wcss.append(y_means.inertia_)
#Plotting WCSS to find the number of clusters
plt.plot(range(1,11), wcss)
plt.xlabel("No. of clusters")
plt.ylabel(" Within Cluster Sum of Squares")
plt.show()
# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(data_health)


#beginning of  the cluster numbering with 1 instead of 0
y_kmeans1=y_kmeans
y_kmeans1=y_kmeans+1

# New Dataframe called cluster
cluster = pd.DataFrame(y_kmeans1)
# Adding cluster to the Dataset1
data_health['cluster'] = cluster
#Mean of clusters
kmeans_mean_cluster = pd.DataFrame(round(data_health.groupby('cluster').mean(),1))
kmeans_mean_cluster
group=1
for group in range(1, 6):
    countries=data_risk.loc[data_risk['Risk_Level']==group]
    listofcoutries= list(countries['country'])
    print(listofcoutries)
