%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

from matplotlib.gridspec import GridSpec

import seaborn as sns

import plotly.express as px

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler as ss
# Data reading and printing head of the data 

data = pd.read_excel("/kaggle/input/online-retail-data-set-from-uci-ml-repo/Online Retail.xlsx")

data.head()
#Checking the shape of the data set

data.shape
#checking the basic information/details of the data

data.info(),data.describe()
# Detailing the Country distribution and customerid

country_data = data[['Country','CustomerID']].drop_duplicates()

country_data.groupby(['Country']).agg({'CustomerID' : 'count'}).sort_values('CustomerID',ascending = False).reset_index().rename(columns = {'CustomerID':'CustomerID Count'})
#Creating a duplicate of the data 

data1 = data.copy()
#filtering out the data set for UK only

data = data[data['Country'] == 'United Kingdom'].reset_index(drop = True)

#data = data.query("Country == 'United Kingdom'")

data.shape
#checking for null values

data.isna().sum()
#Dropping the rows where customerID is missing

data = data[pd.notnull(data['CustomerID'])]



#Checking the description of the data

data.describe()
#filtering data for positive quantity values

data = data.query("Quantity > 0")

data.shape
#Adding new columns as total amount

data['TotalAmount'] = data['UnitPrice']*data['Quantity']
# For recency will check what was the last date of transaction

#First will convert the InvoiceDate as date variable

data['InvoiceDate']=pd.to_datetime(data['InvoiceDate'])

data['InvoiceDate'].max()
#RFM factors calculation:

Latest_date = dt.datetime(2011,12,10)

RFM_data = data.groupby('CustomerID').agg({'InvoiceDate' : lambda x :(Latest_date - x.max()).days,

                                          'InvoiceNo' : 'count','TotalAmount' : 'sum'}).reset_index()



#converting the names of the columns

RFM_data.rename(columns = {'InvoiceDate' : 'Recency',

                          'InvoiceNo' : "Frequency",

                          'TotalAmount' : "Monetary"},inplace = True)

RFM_data.head()
# RFM_data Description/ Summary

RFM_data.iloc[:,1:4].describe()
#Visualizing the Recency, Frequency and Monetary distributions.

i = 0

fig = plt.figure(constrained_layout = True,figsize = (20,5))

gs = GridSpec(1, 3, figure=fig)    



col = ['red','blue','green']

for var in list(RFM_data.columns[1:4]):

    plt.subplot(gs[0,i])

    sns.distplot(RFM_data[var],color= col[i])

    plt.title('Skewness ' + ': ' + round(RFM_data[var].skew(),2).astype(str))

    i= i+1
#Segmentation :

#Here, we will divide the data set into 4 parts based on the quantiles.

quantiles = RFM_data.drop('CustomerID',axis = 1).quantile(q = [0.25,0.5,0.75])

quantiles.to_dict()
#Creating the R,F and M scoring/segement function

#[1] Recency scoring (Negative Impact : Higher the value, less valuable)

def R_score(var,p,d):

    if var <= d[p][0.25]:

        return 1

    elif var <= d[p][0.50]:

        return 2

    elif var <= d[p][0.75]:

        return 3

    else:

        return 4

#[2] Frequency and Monetary (Positive Impact : Higher the value, better the customer)

def FM_score(var,p,d):

    if var <= d[p][0.25]:

        return 4

    elif var <= d[p][0.50]:

        return 3

    elif var <= d[p][0.75]:

        return 2

    else:

        return 1



#Scoring:

RFM_data['R_score'] = RFM_data['Recency'].apply(R_score,args = ('Recency',quantiles,))

RFM_data['F_score'] = RFM_data['Frequency'].apply(FM_score,args = ('Frequency',quantiles,))

RFM_data['M_score'] = RFM_data['Monetary'].apply(FM_score,args = ('Monetary',quantiles,))

RFM_data.head()
#Now we will create : RFMGroup and RFMScore

RFM_data['RFM_Group'] = RFM_data['R_score'].astype(str) + RFM_data['F_score'].astype(str) + RFM_data['M_score'].astype(str)



#Score

RFM_data['RFM_Score'] = RFM_data[['R_score','F_score','M_score']].sum(axis = 1)

RFM_data.head()
#Creating the Customer segments/ Loyality_level

loyalty_level = ['True Lover','Flirting','Potential lover','Platonic Friend']

cuts = pd.qcut(RFM_data['RFM_Score'],q = 4,labels=loyalty_level)

RFM_data['RFM_Loyality_level'] = cuts.values

RFM_data.tail(15)
# Recency V/s Frequency

fig = px.scatter(RFM_data,x = "Recency", y = "Frequency",color = "RFM_Loyality_level")

fig.show()



# Frequency V/s Monetary

fig = px.scatter(RFM_data,x = "Monetary", y = "Frequency",color = "RFM_Loyality_level")

fig.show()



# Monetary V/s Recency

fig = px.scatter(RFM_data,x = "Monetary", y = "Recency",color = "RFM_Loyality_level")

fig.show()
# First will focus on the negativ and zero before the transformation.

def right_treat(var):

    if var <= 0:

        return 1

    else:

        return var



# Describing the data

RFM_data.describe()
#Applying on the data.

RFM_data['Recency'] = RFM_data['Recency'].apply(lambda x : right_treat(x))

RFM_data['Monetary'] = RFM_data['Monetary'].apply(lambda x : right_treat(x))



#Checking the Skewness of R, F and M

print('Recency Skewness : ' + RFM_data['Recency'].skew().astype(str))

print('Frequency Skewness : ' + RFM_data['Frequency'].skew().astype(str))

print('Monetary Skewness : ' + RFM_data['Monetary'].skew().astype(str))
#Log Transformation

log_RFM_data = RFM_data[['Recency','Frequency','Monetary']].apply(np.log,axis = 1).round(4)
#Plot after transformation for the distributions :

i = 0

fig = plt.figure(constrained_layout = True,figsize = (20,5))

gs = GridSpec(1, 3, figure=fig)    



col = ['red','blue','green']

for var in list(log_RFM_data.columns[0:3]):

    plt.subplot(gs[0,i])

    sns.distplot(log_RFM_data[var],color= col[i])

    plt.title('Skewness ' + ': ' + round(log_RFM_data[var].skew(),2).astype(str))

    i= i+1

log_RFM_data.describe()
#Scaling the data

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

Scaled_RFM_data = ss.fit_transform(log_RFM_data)

Scaled_RFM_data = pd.DataFrame(Scaled_RFM_data,columns=log_RFM_data.columns,index=log_RFM_data.index)
# Will search the optimal number of cluster based on the Elbow Method as below:

SS_distance = {}

for k in range(1,20):

    mod = KMeans(n_clusters= k, max_iter=1000,init = 'k-means++')

    mod = mod.fit(Scaled_RFM_data)

    SS_distance[k] = mod.inertia_



#Plotting the sum of square distance values and numbers of clusters

plt.figure(figsize = (15,5))

sns.pointplot(x = list(SS_distance.keys()), y = list(SS_distance.values()))

plt.xlabel("Number of clusters")

plt.ylabel("Sum of square Distances")

plt.title("Elbow Techinque to find the optimal cluster size")
# Now we will perform K- means clustering on the data set.

KM_clust = KMeans(n_clusters= 3, init = 'k-means++',max_iter = 1000)

KM_clust.fit(Scaled_RFM_data)



# Mapping on the data

RFM_data['Cluster'] = KM_clust.labels_

RFM_data['Cluster'] = 'Cluster' + RFM_data['Cluster'].astype(str)

RFM_data.head()
# Recency V/s Frequency

fig = px.scatter(RFM_data,x = 'Recency',y = 'Frequency', color = 'Cluster')

fig.show()



# Frequency V/s Monetary

fig = px.scatter(RFM_data,x = 'Monetary',y = 'Frequency', color = 'Cluster')

fig.show()



# Recency V/s Monetary

fig = px.scatter(RFM_data,x = 'Monetary',y = 'Recency', color = 'Cluster')

fig.show()
