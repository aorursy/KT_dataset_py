# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from kmodes.kmodes import KModes

import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load into pandas dataframe
us_data = pd.read_csv("/kaggle/input/us-police-shootings/shootings.csv")
us_data.head()
# Copy of US data
us_data_tr = us_data

# Keep important features
us_data_tr = us_data_tr.drop(['id','name','manner_of_death','armed','gender','city','state','body_camera','arms_category'],axis=1)

# Convert categorical variables to numeric
us_data_tr = pd.get_dummies(us_data_tr, columns=['race','signs_of_mental_illness','threat_level','flee'])
#loop over entries for producing histograms and graphs etc.

#easiest initial plot will be a histogram for all months not separating year. This will tell us what month the police on average shoot the most people. Is it christmas because of all the christmas spirit?

month_list=[]
yearlist=[]
for i in range(len(us_data_tr)):
    year=float(us_data_tr['date'][i].split("-")[0])
    month=float(us_data_tr['date'][i].split("-")[1])
    day=us_data_tr['date'][i].split("-")[2]
    month_list.append(month)
    yearlist.append(year)
fig1=plt.figure()
monthHist=plt.hist(month_list,12,facecolor='r',label='Month')
plt.title("month")
plt.ylabel('shootings per month')
plt.xlabel('month')
fig2=plt.figure()
YearHist=plt.hist(yearlist,5,facecolor='g',label='Year')
plt.title("year")
plt.ylabel('shootings per year')
plt.xlabel('Year')
plt.show()
#now drop the date as it causes an error later on 
us_data_tr = us_data_tr.drop(['date'],axis=1)

#Note that in 2019 the number of shootings was roughly 1200, an increase of roughly 300 fromthe year before. 
#The poisson fluctuations expected are sqrt(1200)~34 meaning we can conclude this increase is not from random statistical fluctuations. 
#This begs the question, why did the shootings increase so much that year?
# Standardize
scaler = MinMaxScaler()
us_data_std = scaler.fit_transform(us_data_tr)
# Check K-mode optimal cluster
cost = []
K = 5
for num_clusters in list(range(1,K)):
    kmode = KModes(n_clusters=num_clusters, init = "Huang", n_init = 1, verbose=1)
    kmode.fit_predict(us_data_std)
    cost.append(kmode.cost_)
k = np.array([i for i in range(1,K,1)])
plt.xlabel('number of clusters')
plt.ylabel('cost')
plt.plot(k,cost)
# K-mode clustering
kmode = KModes(n_clusters=3, init = "Huang", n_init = 1, verbose=1)
clusters = kmode.fit_predict(us_data_std)
# Add cluster number to new dataframe
clusters = pd.DataFrame(clusters)
clusters.columns = ['cluster']
us_data_clust = pd.concat([us_data, clusters], axis = 1).reset_index()
# Plot race clusters
plt.subplots(figsize = (15,5))
sns.countplot(x=us_data_clust['race'],order=us_data_clust['race'].value_counts().index,hue=us_data_clust['cluster'])
plt.show()
# Plot mental illness clusters
plt.subplots(figsize = (15,5))
sns.countplot(x=us_data_clust['signs_of_mental_illness'],order=us_data_clust['signs_of_mental_illness'].value_counts().index,hue=us_data_clust['cluster'])
plt.show()