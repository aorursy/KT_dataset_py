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
import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as stats

from sklearn import ensemble, tree, linear_model, preprocessing

import missingno as msno

import pandas_profiling

import plotly.express as px
District_wise = pd.read_csv('../input/education-in-india/2015_16_Districtwise.csv')

State_wise_elementry = pd.read_csv('../input/education-in-india/2015_16_Statewise_Elementary.csv')

State_wise_secondary = pd.read_csv('../input/education-in-india/2015_16_Statewise_Secondary.csv')

District_wise_met = pd.read_csv('../input/education-in-india/2015_16_Districtwise_Metadata.csv')

State_wise_elementry_met = pd.read_csv('../input/education-in-india/2015_16_Statewise_Elementary_Metadata.csv')

State_wise_secondary_met = pd.read_csv('../input/education-in-india/2015_16_Statewise_Secondary_Metadata.csv')

District_wise.head()
District_wise_met.head()
District_wise_total = pd.DataFrame()
i=0

for name in District_wise_met['Description']:

    if 'Total' in name:

        District_wise_total[District_wise_met.iloc[i][1]] = District_wise[District_wise_met.iloc[i][0]]

    i=i+1
District_wise_total['Schools_By_Category: Total']/(District_wise_total['Schools_by_Category:_Government: Total']+District_wise_total['Schools_by_Category:_Private_: Total']+District_wise_total['Schools_by_Category:_Madarsas_&_Unrecognised: Total'])
District_wise.head()
District_wise_new = pd.DataFrame()
District_wise_new['STATNAME'] = District_wise['STATNAME']
District_wise_new['DISTNAME'] = District_wise['DISTNAME']
District_wise_new = pd.concat([District_wise_new, District_wise_total], axis = 1 )
District_wise_grouped = District_wise_new.groupby(by = 'STATNAME')
State_wise_sum = District_wise_grouped.sum()
State_wise_sum.head()
State_wise_sum.index
State_wise_sum['People_per_School'] = State_wise_sum['Basic_data_from_Census_2011: Total_Population(in_1000\'s)']/State_wise_sum['Schools_By_Category: Total']
ax = sns.barplot(y=State_wise_sum.index, x='People_per_School', data = State_wise_sum  )

#ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

#plt.tight_layout()



plt.figure(figsize=(16,4))
State_wise_sum.head()
State_wise_fac = pd.DataFrame()

for name in State_wise_sum.columns[21:31]:

    

    State_wise_fac[name] = State_wise_sum[name]
State_wise_fac['Total_Schools'] = State_wise_sum['Schools_By_Category: Total']

State_wise_fac['Population_school_serves'] = State_wise_sum['People_per_School']
for name in State_wise_fac.columns[0:8]:

    State_wise_fac[name] = State_wise_fac[name]/State_wise_fac['Total_Schools']
State_wise_fac['Fraction_of_required_ramps'] = State_wise_fac['Schools_with_Ramp_(where_needed): Total']/State_wise_fac['Schools_where_Ramp_is_Required: Total']
State_wise_fac = State_wise_fac.drop(['Schools_with_Ramp_(where_needed): Total','Schools_where_Ramp_is_Required: Total'], axis =1)
State_wise_fac.head()
pd.plotting.scatter_matrix(State_wise_fac, alpha = 0.3, figsize = (21,12), diagonal = 'kde')
df_std
df_std[:,1]
State_wise_fac_scaled = pd.DataFrame()

i=0

for name in State_wise_fac.columns:

    State_wise_fac_scaled[name] = df_std[:,i]

    i=i+1
State_wise_fac.head()
temp = State_wise_fac

temp['Schools_with_Computer: Total'] = np.log(State_wise_fac['Schools_with_Computer: Total'])

temp['Total_Schools'] = np.log(State_wise_fac['Total_Schools'])

temp['Population_school_serves'] =  np.log(State_wise_fac['Population_school_serves'])
pd.plotting.scatter_matrix(temp, alpha = 0.3, figsize = (21,12), diagonal = 'kde')
temp.head()
temp.drop(['Schools_with_Girls\'_Toilet: Total','Schools_with_Drinking_Water: Total'], inplace = True, axis=1)
temp['Schools_with_Boys\'_Toilet: Total'] = np.arcsin(temp['Schools_with_Boys\'_Toilet: Total'])
std_scale = preprocessing.StandardScaler().fit(temp)

df_std = std_scale.transform(temp)

df_std
temp = pd.DataFrame()

i=0

for name in State_wise_fac.columns:

    temp[name] = df_std[:,i]

    i=i+1
temp.head()
pd.plotting.scatter_matrix(temp, alpha = 0.3, figsize = (21,12), diagonal = 'kde')
plt.subplots(figsize=(12,9))

sns.heatmap(abs(temp.cov()), vmax=0.9)
from sklearn.decomposition import PCA

pca = PCA(n_components=temp.shape[1]).fit(temp)

#Fitting the PCA algorithm with our Data

pca = PCA().fit(temp)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

#plt.title('Pulsar Dataset Explained Variance')

plt.show()
pca = PCA(n_components=5)

transformed_temp = pca.fit_transform(temp)
transformed_temp[:,1]
for_train = pd.DataFrame()

i=0

for name in ['1','2','3','4','5']:

    for_train[name] = transformed_temp[:,i]

    i=i+1
for_train.head()
from sklearn.cluster import KMeans 

from sklearn import metrics 

from scipy.spatial.distance import cdist 



distortions = [] 

inertias = [] 

mapping1 = {} 

mapping2 = {} 

K = range(1,10) 

  

for k in K: 

    #Building and fitting the model 

    kmeanModel = KMeans(n_clusters=k).fit(for_train) 

    kmeanModel.fit(for_train)     

      

    distortions.append(sum(np.min(cdist(for_train, kmeanModel.cluster_centers_, 

                      'euclidean'),axis=1)) / for_train.shape[0]) 

    inertias.append(kmeanModel.inertia_) 

  

    mapping1[k] = sum(np.min(cdist(for_train, kmeanModel.cluster_centers_, 

                 'euclidean'),axis=1)) / for_train.shape[0] 

    mapping2[k] = kmeanModel.inertia_ 
for key,val in mapping1.items(): 

    print(str(key)+' : '+str(val)) 


plt.plot(K, distortions, 'bx-') 

plt.xlabel('Values of K') 

plt.ylabel('Distortion') 

plt.title('The Elbow Method using Distortion') 

plt.show() 