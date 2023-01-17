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
District_wise_total = pd.DataFrame()
i=0

for name in District_wise_met['Description']:

    if 'Total' in name:

        District_wise_total[District_wise_met.iloc[i][1]] = District_wise[District_wise_met.iloc[i][0]]

    i=i+1
District_wise_total.info()
District_wise_fac = pd.DataFrame()

for name in District_wise_total.columns[21:31]:

    

    District_wise_fac[name] = District_wise_total[name]
District_wise_fac.head()
District_wise_fac['Fraction_of_ramp'] = District_wise_fac['Schools_with_Ramp_(where_needed): Total']/District_wise_fac['Schools_where_Ramp_is_Required: Total']
District_wise_fac.drop(['Schools_with_Ramp_(where_needed): Total','Schools_where_Ramp_is_Required: Total'], axis=1, inplace=True)
District_wise_fac.head()
District_wise_fac['Population_per_school'] = District_wise_total['Basic_data_from_Census_2011: Total_Population(in_1000\'s)']/District_wise_total['Schools_By_Category: Total']
District_wise_fac.drop('Schools_with_Boys\'_Toilet: Total', axis=1, inplace = True)
pd.plotting.scatter_matrix(District_wise_fac, alpha = 0.3, figsize = (21,12), diagonal = 'kde')


Ramp = District_wise_fac['Fraction_of_ramp']
District_wise_fac.drop('Fraction_of_ramp', axis=1, inplace=True)
for name in District_wise_fac.columns:

    District_wise_fac[name]=np.log(District_wise_fac[name])
Ramp = np.arcsin(Ramp)
sns.stripplot(Ramp)
District_wise_fac['Fraction_needed_ramp_give'] = Ramp
pd.plotting.scatter_matrix(District_wise_fac, alpha = 0.3, figsize = (21,12), diagonal = 'kde')
std_scale = preprocessing.StandardScaler().fit(District_wise_fac)

df_std = std_scale.transform(District_wise_fac)

df_std
df_std[:,0]
District_wise_scaled = pd.DataFrame()


i=0

for name in District_wise_fac.columns:

    District_wise_scaled[name] = df_std[:,i]

    i=i+1
District_wise_scaled.head()
pd.plotting.scatter_matrix(District_wise_scaled, alpha = 0.3, figsize = (21,12), diagonal = 'kde')
District_wise_scaled.describe()
District_wise_scaled = District_wise_scaled[District_wise_scaled['Population_per_school'].notna()]
sns.heatmap(District_wise_scaled.corr())
#Fitting the PCA algorithm with our Data

pca = PCA().fit(District_wise_scaled)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

#plt.title('Pulsar Dataset Explained Variance')

plt.show()
pca = PCA(n_components=2)

transformed_temp = pca.fit_transform(District_wise_scaled)
for_train = pd.DataFrame()

i=0

for name in ['1','2',]:

    for_train[name] = transformed_temp[:,i]

    i=i+1
for_train.head()
plt.matshow(pca.components_,cmap='viridis')

plt.colorbar()

plt.xticks(range(len(District_wise_scaled.columns)),District_wise_scaled.columns,rotation=65,ha='left')

plt.tight_layout()

plt.show()#

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
kmeanModel = KMeans(n_clusters=4).fit(for_train) 

kmeanModel.fit(for_train)  
y_kmeans = kmeanModel.predict(for_train)
plt.scatter(for_train['1'], for_train['2'], c=y_kmeans, s=50)

plt.xlabel('Component 1')

plt.ylabel('Component 2')
#Code for 3D Plot.



from mpl_toolkits.mplot3d import Axes3D

threedee = plt.figure().gca(projection='3d')



threedee.scatter(for_train['1'], for_train['2'], for_train['3'], c=y_kmeans, s=50, cmap='viridis', label = label)



threedee.set_xlabel('Component 1')

threedee.set_ylabel('Component 2')

threedee.set_zlabel('Component 3')

plt.show()
District_wise = District_wise[District_wise['TOTPOPULAT'].notna()]
i=0

j=0

xaxis=[]

yaxis=[]

cc=[]

for state in District_wise['STATNAME'] :

    if state ==( 'BIHAR'):

        xaxis.append(for_train['1'][i])

        yaxis.append(for_train['2'][i])

        cc.append(y_kmeans[i])

    i=i+1

        

        

fig, ax = plt.subplots()



scatter = ax.scatter(xaxis, yaxis, c=cc, s=50)



legend1 = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")

ax.add_artist(legend1)

ax.grid(True)





plt.xlabel('Component 1')

plt.ylabel('Component 2') 



plt.show()
i=0

j=0

xaxis=[]

yaxis=[]

cc=[]

for state in District_wise['STATNAME'] :

    if state == 'TAMIL NADU':

        xaxis.append(for_train['1'][i])

        yaxis.append(for_train['2'][i])

        cc.append(y_kmeans[i])

    i=i+1

        

        

plt.scatter(xaxis, yaxis, c=cc, s=50)

plt.xlabel('Component 1')

plt.ylabel('Component 2')   