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
ncov = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')

print(ncov.head(7))

print('\n')
print(ncov.info())
ncov = ncov.drop('Sno', axis = 1)

ncov.columns = ['State', 'Country', 'Date', 'Confirmed', 'Deaths', 'Recovered']

ncov['Date'] = ncov['Date'].apply(pd.to_datetime).dt.normalize() #convert to proper datetime object
ncov.info()
ncov[['State','Country','Date']].drop_duplicates().shape[0] == ncov.shape[0]
ncov.describe(include = 'all')
ncov[['Country','State']][ncov['State'].isnull()].drop_duplicates()
ncov[ncov['Country'].isin(list(ncov[['Country','State']][ncov['State'].isnull()]['Country'].unique()))]['State'].unique()
ncov.State.unique()
ncov.Country.unique()
print(ncov[ncov['Country'].isin(['China', 'Mainland China'])].groupby('Country')['State'].unique())

print(ncov[ncov['Country'].isin(['China', 'Mainland China'])].groupby('Country')['Date'].unique())
ncov['Country'] = ncov['Country'].replace(['Mainland China'], 'China') #set 'Mainland China' to 'China'

sorted(ncov.Country.unique())
print(ncov.head())
china = ncov[ncov['Country']=='China']

china.head()
import matplotlib.pyplot as plt

%matplotlib inline



plt.rcParams["figure.figsize"] = (7,5)

ax1 = china[['Date','Confirmed']].groupby(['Date']).sum().plot()

ax1.set_ylabel("Total Number of Confirmed Cases")

ax1.set_xlabel("Date")



ax2 = china[['Date','Deaths', 'Recovered']].groupby(['Date']).sum().plot()

ax2.set_ylabel("Total N")

ax2.set_xlabel("Date")

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import seaborn as sns

plt.rcParams["figure.figsize"] = (17,10)

nums = china.groupby(["State"])['Confirmed'].aggregate(sum).reset_index().sort_values('Confirmed', ascending= False)

ax = sns.barplot(x="Confirmed", y="State", order = nums['State'], data=china, ci=None) 

ax.set_xlabel("Total Confirmed Cases")
#a custom function to return the lower and upper bounds of 95% confidence interval of a proportion

def get_ci(N,p):

    lci = (p - 1.96*(((p*(1-p))/N) ** 0.5))*100

    uci = (p + 1.96*(((p*(1-p))/N) ** 0.5))*100

    return str(np.round(lci,3)) + "% - " + str(np.round(uci,3)) + '%'
final = ncov[ncov.Date==np.max(ncov.Date)]

final = final.copy()



final['CFR'] = np.round((final.Deaths.values/final.Confirmed.values)*100,3)

final['CFR 95% CI'] = final.apply(lambda row: get_ci(row['Confirmed'],row['CFR']/100),axis=1)

global_cfr = np.round(np.sum(final.Deaths.values)/np.sum(final.Confirmed.values)*100, 3)

final.sort_values('CFR', ascending= False).head(10)
tops = final.sort_values('CFR', ascending= False)

tops = tops[tops.CFR >0]

df = final[final['CFR'] != 0]

plt.rcParams["figure.figsize"] = (10,5)

ax = sns.barplot(y="CFR", x="State", order = tops['State'], data=df, ci=None) 

ax.axhline(global_cfr, alpha=.5, color='r', linestyle='dashed')

ax.set_title('Case Fatality Rates (CFR) as of 30 Jan 2020')

ax.set_ylabel('CFR %')

print('Average CFR % = ' + str(global_cfr))
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import LocalOutlierFactor

scaler = StandardScaler()

scd = scaler.fit_transform(final[['Confirmed','Deaths','Recovered']])

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1) #LOF is very sensitive to the choice of n_neighbors. Generally, n_neighbors = 20 works better

clf.fit(scd)

lofs = clf.negative_outlier_factor_*-1

final['LOF Score'] = lofs

tops = final.sort_values('LOF Score', ascending= False)

plt.rcParams["figure.figsize"] = (20,12)

ax = sns.barplot(x="LOF Score", y="State", order = tops['State'], data=final, ci=None) 

ax.axvline(1, alpha=.5, color='g', linestyle='dashed')

ax.axvline(np.median(lofs), alpha=.5, color='b', linestyle='dashed')

ax.axvline(np.mean(lofs) + 3*np.std(lofs), alpha=.5, color='r', linestyle='dashed')
final.sort_values('LOF Score', ascending=False)
from sklearn.cluster import KMeans

plt.rcParams["figure.figsize"] = (5,5)

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=1897)

    kmeans.fit(scd)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('Within Cluster Sum of Squares')

plt.show()
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=1897)

clusters = np.where(kmeans.fit_predict(scd) == 0, 'Cluster 1', 'Cluster 2')

clusters
from sklearn import decomposition

pca = decomposition.PCA(n_components=3)

pca.fit(scd)

X = pca.transform(scd)

print(pca.explained_variance_ratio_.cumsum())
plt.rcParams["figure.figsize"] = (7,7)

ax = sns.scatterplot(X[:,0], X[:,1], marker = 'X', s = 80, hue=clusters)

ax.set_title('K-Means Clusters of States/Provinces')

ax.set_xlabel('Principal Component 1')

ax.set_ylabel('Principal Component 2')
pd.DataFrame(final.State.values, clusters)