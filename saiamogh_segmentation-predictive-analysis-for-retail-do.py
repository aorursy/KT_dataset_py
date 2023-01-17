# Import Packages



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.preprocessing import scale,StandardScaler

from sklearn.metrics import silhouette_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
# Loading the data into dataframe

df = pd.read_csv('/kaggle/input/online-retail-ii-uci/online_retail_II.csv',sep=',',

    header='infer')
df.head(5)
# Removing Missing values based on customer id 



print(df.isnull().sum())

df.dropna(subset=['Customer ID'],inplace= True)

df.head()
# Unique values for each column

df.nunique().reset_index(name ='Unique Values')
# Country distribution by %



country_count = df['Country'].value_counts().sort_values(ascending = False).reset_index(name = 'Count').rename(columns = {'index':'Country'})



country_count['%'] = country_count['Count'].div(np.sum(country_count['Count']))*100



country_count['Major Country Name'] = country_count[['%','Country']].apply( lambda x : 'Others' if( x['%'] < 1) else x['Country'],axis=1)



sns.barplot(y=country_count['Major Country Name'],x=country_count['%'])
# Price Distribution



sns.distplot(df['Price'],bins = 50,rug=True,hist=False)

# Price range is very varied and there some extream values in prices, So we have to do normalization to reduce the varied effect.
df[['Quantity','Price']].describe()
# Looks like there -ne quantity values, they may be refunds or data quality issue, I am discarding them for now



df = df[df['Quantity'] > 0] 
df[['Quantity','Price']].describe()
sns.boxplot(x=df['Price'])
df[(df['Price'] >= 500) & (df['Price'] <= 1000)]['Description'].unique()
# Droping the above Description variables can be one options but this will lead to data loss



df['Revenue'] = df['Quantity'] * df['Price']

df_ = df[df['InvoiceDate'] < '2010-12-01']

RFM = pd.DataFrame()

max_date = pd.to_datetime(max(df_['InvoiceDate']))

RFM[['Customer ID','Recency','Frequency','Monetary Value']] = df_[['Customer ID','Revenue','Invoice','InvoiceDate']].groupby('Customer ID').agg({'InvoiceDate':'min','Invoice':'nunique','Revenue':'sum'}).reset_index().rename(columns={'Invoice':'Frequency'})

RFM['Recency'] = (max_date - pd.to_datetime(RFM['Recency'])).dt.days
quantiles = RFM.quantile(q=[0.20,0.4,0.6,0.8])
def recency(data,column):

    rule =[]

    for row in data[column]:

        if row <= quantiles[column][0.2]:

            rule.append(5)

        elif  row <=  quantiles[column][0.4]:

            rule.append(4)

        elif row <=  quantiles[column][0.6]:

            rule.append(3) 

        elif  row <=  quantiles[column][0.8]:

            rule.append(2)

        else:

            rule.append(1)



    return rule



def fm(data,column):

    rule = []

    for row in data[column]:

        if row <= quantiles[column][0.2]:

            rule.append(1)

        elif  row <= quantiles[column][0.4]:

            rule.append(2)

        elif row <= quantiles[column][0.6]:

            rule.append(3)

        elif row <= quantiles[column][0.8]:

            rule.append(4)   

        else:

            rule.append(5)



    return rule
RFM['Recency Score'] = recency(data=RFM ,column='Recency')

RFM['Frequency Score'] = fm(data=RFM ,column='Frequency')

RFM['Monetary Score'] = fm(data=RFM ,column='Monetary Value')
RFM['Score'] = RFM['Recency Score'].map(str) + RFM['Frequency Score'].map(str) + RFM['Monetary Score'].map(str)
RFM.head(2)
RFM_norm = scale(RFM.iloc[:,1:])
Elbow ={}

silhouette ={}

for k in range(2,11):

    k_mean = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=1000, tol=0.0001)

    k_mean.fit(RFM_norm)

    lables = k_mean.labels_

    Elbow[k] = k_mean.inertia_

    silhouette[k] = silhouette_score(RFM_norm,lables,metric='euclidean')
Elbow
# k vs inertia(WSS)

sns.lineplot(x = [i for i in Elbow.keys()] ,y =[i for i in Elbow.values()])
# Elbow method :- Calculates the within-cluster-sum-of-squares error (Total Variance for each k value )

# k = 3 | 4 is unclear so we need to look at another metric called silhouette measure

# Silhouette measure (b - a) / max(a, b) :- Calculates how similar a point is within the cluster and to other clusters [-1,+1]
# k vs silhouette

sns.lineplot(x = [i for i in silhouette.keys()] ,y =[i for i in silhouette.values()])
# from the above we can deside that 3 clusters are optimal for this dataset
k_mean = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=1000, tol=0.0001)

k_mean.fit(RFM_norm)

y = k_mean.predict(RFM_norm)

RFM['Cluster'] = y

RFM['Cluster Names'] = y
RFM[RFM['Cluster'] ==0].iloc[:,1:].describe()
RFM[RFM['Cluster'] == 1].iloc[:,1:].describe()
RFM[RFM['Cluster'] == 2].iloc[:,1:].describe()
segt_map = {

    r'[1-2][1-2][1-5]': 'Hibernating',

    r'[1-2][3-4][1-5]': 'At risk',

    r'[1-2]5[1-5]': 'Can\'t loose',

    r'3[1-2][1-5]': 'About to sleep',

    r'33[1-5]': 'Need attention',

    r'[3-5][4-5][1-5]': 'Loyal customers',

    r'41[1-5]': 'Active one timers ',

    r'51[1-5]': 'New customers',

    r'[4-5][2-3][1-5]': 'Potential loyalists',

    r'5[4-5][4-5]': 'Champions',

    r'4[2-5][1-5]': 'Active Customers',

   

}





RFM['Sub Segment'] = RFM['Score'].replace(segt_map, regex=True)


RFM.replace({'Cluster Names':{2:'High FM & High Recency',1:'Avg FM & High Recency',0:'Low FM & Low Recency'}},inplace=True)

# Predictive analytics:- we can extend the cluster analytics for prediction by adding few customer dimentions
customer_dim= df[['Customer ID','Quantity','Country']].groupby('Customer ID').agg({'Quantity':'sum','Country':'unique'}).reset_index()

RFM = RFM.merge(customer_dim[['Customer ID','Quantity','Country']])

RFM['Country']= RFM['Country'].apply(lambda x : str(x).strip('[]'))
Label_e = LabelEncoder()

RFM['Country_Norm'] = Label_e.fit_transform(RFM['Country'])

RFM['Score_Norm'] = Label_e.fit_transform(RFM['Score'])

RFM['Sub_Segment_Norm'] = Label_e.fit_transform(RFM['Sub Segment'])
X = RFM[['Recency', 'Frequency', 'Monetary Value',

       'Recency Score', 'Frequency Score','Monetary Score','Country_Norm', 'Score_Norm','Sub_Segment_Norm']]

y = RFM['Cluster']
decision_t = DecisionTreeClassifier()

fold = cross_val_score(estimator=decision_t,X=X,y=y,cv=3)

print(fold.mean())