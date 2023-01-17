'''



Context

Computer Network Traffic Data - A ~500K CSV with summary of some real network traffic data from the past. The dataset has ~21K rows and covers 10 local workstation IPs over a three month period. Half of these local IPs were compromised at some point during this period and became members of various botnets.



Content

Each row consists of four columns:



date: yyyy-mm-dd (from 2006-07-01 through 2006-09-30)

l_ipn: local IP (coded as an integer from 0-9)

r_asn: remote ASN (an integer which identifies the remote ISP)

f: flows (count of connnections for that day)

Reports of "odd" activity or suspicions about a machine's behavior triggered investigations on the following days (although the machine might have been compromised earlier)



Date : IP 08-24 : 1 09-04 : 5 09-18 : 4 09-26 : 3 6





'''
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from matplotlib import pyplot as plt

from scipy.stats import zscore
df = pd.read_csv('../input/cs448b_ipasn.csv')
df.head()
dfOrig = df.copy()

#df = dfOrig.copy()
df.describe()
df.info()
df.columns
#vivewing given examples 
df[(df.l_ipn == 1) & (df.date == '2006-08-24')]

df[(df.l_ipn == 5) & (df.date == '2006-09-04')]



 
df[(df.l_ipn == 4) & (df.date == '2006-09-18')]
df[(df.l_ipn == 3) & (df.date == '2006-09-26')]
#removing f == 1

#df = df[df.f > 1]
#len(df)/len(dfOrig)
df.l_ipn.value_counts()
# 0 is he most active user, 3 is the least
for ip in set(df.l_ipn):

    fNormed = df.loc[(df.l_ipn == ip),'f']

    plt.boxplot(fNormed,len(fNormed) * [0],".")

    plt.title('IP:' + str(ip))

    plt.show()
for ip in set(df.l_ipn):

    df[df.l_ipn == ip].f.hist(bins = 100)

    plt.autoscale(enable=True, axis='both', tight=None)

    plt.title(('IP: %d') % ip)

    plt.show()
# instead use log scale since anomaly detection (have skewness of large values for "normal activity")



for ip in set(df.l_ipn):

    df[df.l_ipn == ip].f.hist(log=True,bins =200)

    plt.title(('IP: %d') % ip)

    plt.show()
#normalize flows per IP since different ratios and scales 
#sort by IP address

df.sort_values(inplace=True, by=['l_ipn'])
from sklearn.preprocessing import robust_scale

# Scale features using statistics that are robust to outliers.

# using robust_scale instead of RobustScaler since want to Standardize a dataset along any axis
from sklearn.preprocessing import StandardScaler
#accessing columns



#sample[sample['l_ipn'] == 0].fNorm = 2 # not good. returns a copy of sample.l_ipn



#use iloc or loc instead 

# .loc[criterion,selection]

#use df.iloc[1, df.columns.get_loc('s')] = 'B'    or use df.loc[df.index[1], 's'] = 'B'



#sample[sample.iloc[:,sample.columns.get_loc('l_ipn')] == 0]

#sample[sample.loc[:,'l_ipn'] == 0]
#normalize traffic for each IP 



#scaler = robust_scale()

scaler = StandardScaler()
for ip in set(df.l_ipn):

    df.loc[(df.l_ipn == ip),'fNorm'] = scaler.fit_transform(df.loc[(df.l_ipn == ip),'f'].values.reshape(-1, 1)) # reshaped since it's scaling a single feature only 

    df.loc[(df.l_ipn == ip),'fMean'] = scaler.mean_

    df.loc[(df.l_ipn == ip),'fVar'] = scaler.var_
for ip in set(df.l_ipn):

    fNormed = df.loc[(df.l_ipn == ip),'fNorm']

    plt.plot(fNormed,len(fNormed) * [0],".")

    fMean = df.loc[(df.l_ipn == ip),'fMean'].iloc[0]# only need the first value as they are all the same in this column for this ip

    plt.plot(fMean,0,'ro')

    plt.title('IP:' + str(ip))

    plt.show()
# it is clear that there are anomalies in the amount of traffic flow 
# todo: trying a scaler for skewed data
#analyzing ASN
#is asn unique per user?
listOfAsnsPerUser = [[]] * len(df.l_ipn)
numAsnsPerIp = 0

for ip in set(df.l_ipn):

    numAsnsPerIp += len(set(df.loc[(df.l_ipn == ip),'r_asn']))
numAsnsPerIp
len(set(df.loc[:,'r_asn']))
#number of  unique of asns per ip != number of unqiue asns for total dataset

#therefore, asns are not unique per IP
#using asns as categorical variable 
dfDummy = df.copy()
dfDummy = pd.get_dummies(df,columns=['r_asn'],drop_first=True)
dfDummy.head()
#takes too long 

# dfDummy.drop(labels =['date','fNorm','fMean','fVar','l_ipn'],axis=1).corr() 
dfCorrAsnFlow = dfDummy.drop(labels =['date','fNorm','fMean','fVar','l_ipn'],axis=1)
# todo: look for anomalies in users using suddently a different ASN and have a high traffic flow 
# time sampling
df.head()
df.date = pd.to_datetime(df.date,errors='coerce')
len(df.date) == len(df.date.dropna()) 
# all dates were valid 
df.info()
#df.date.hist(bins = 100)
#fig = plt.figure(figsize = (15,20))

#ax = fig.gca()

#df.date.hist(bins = 50, ax = ax)
df = df.sort_values('date', ascending=True)

plt.figure(figsize=(15,20))

plt.plot(df['date'], df['f'])

plt.xticks(rotation='vertical')
# now per ip 
for ip in set(df.l_ipn):

    plt.figure(figsize=(10,15))

    plt.xticks(rotation='vertical')

    plt.title(('IP: %d') % ip)

    plt.plot(df[df.l_ipn == ip]['date'], df[df.l_ipn == ip]['f'])

    plt.show()
#using IP = 4 as poc 
dataset = df.copy()
dataset = pd.get_dummies(dataset,columns=['r_asn'],drop_first=True)
dataset.head()
dataset = dataset.drop(labels =['date','fNorm','fMean','fVar'],axis=1)
dataset.head()
#using IP == 4 as POC
dataset = dataset[dataset.l_ipn == 4].drop(['l_ipn'],axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dataset.loc[:,'f'] = scaler.fit_transform(dataset.f.values.reshape(-1, 1))
dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(dataset)
kmeans.labels_
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
centroids = kmeans.cluster_centers_
centroids
centroids2d = pd.DataFrame(pca.fit_transform(centroids))
centroids2d
xPca = centroids2d.loc[:,0]

yPca = centroids2d.loc[:,1]
xPca
yPca
plt.scatter(xPca,yPca)
# will create a single dataset without any anomalies for IP = 4 
plt.figure(figsize=(10,15))

plt.xticks(rotation='vertical')

plt.yscale('log')

plt.title(('IP: %d') % 4)

plt.plot(range(len(dataset)), dataset['f'])

plt.show()
len(dataset[(dataset.f < 10**-.5)])
len(dataset)
#seems as those two days were anomalious
negativeClass = dataset[(dataset.f >= 10**-.5)]
negativeClass
positiveClass = dataset.head(2)
positiveClass
#removing test classes from dataset

dataset = dataset.drop(dataset.index[[0,1]])
len(dataset)
dataset = dataset[(dataset.f < 10**-.5)]
len(dataset)
kmeans.predict(positiveClass)
kmeans.predict(negativeClass)
posRes = kmeans.transform(positiveClass)
negRes = kmeans.transform(negativeClass)
negRes[0]
from numpy import linalg
centroids
dist = numpy.linalg.norm(a-b)
# todo checkout auto encoders
#todo checkout one class SVM
#create classifier and split data into train test and validation 