import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns

import numpy as np

plt.style.use('fivethirtyeight')

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import folium

import folium.plugins

from matplotlib import animation,rc

import plotly.io as pio

import io



import base64

from IPython.display import HTML, display

import codecs

from subprocess import check_output



print(check_output(["ls", "../input"]).decode("utf8"))

terror=pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')



terror.head()
print (terror.describe())
terror.isnull().sum()
terror.loc[:, terror.isna().any()]

def missing_zero_values_table(terror):

        zero_val = (terror == 0.00).astype(int).sum(axis=0)

        mis_val = terror.isnull().sum()

        mis_val_percent = 100 * terror.isnull().sum() / len(terror)

        mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)

        mz_table = mz_table.rename(

        columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})

        mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']

        mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(terror)

        mz_table['Data Type'] = terror.dtypes

        mz_table = mz_table[

            mz_table.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        print ("Your selected dataframe has " + str(terror.shape[1]) + " columns and " + str(terror.shape[0]) + " Rows.\n"      

            "There are " + str(mz_table.shape[0]) +

              " columns that have missing values.")

#         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)

        return mz_table



missing_zero_values_table(terror)
df= terror.drop(['approxdate','resolution','addnotes','scite1','scite2','scite3','weaptype2',

'weaptype2_txt',

'weapsubtype2',

'weapsubtype2_txt',

'weaptype3',

'weaptype3_txt',

'weapsubtype3',

'weapsubtype3_txt',

'weaptype4',

'weaptype4_txt',

'weapsubtype4',

'weapsubtype4_txt',

'nperpcap',

'claimed',

'claimmode',

'claimmode_txt',

'claim2',

'claimmode2',

'claimmode2_txt',

'claim3',

'claimmode3',

'claimmode3_txt',

'compclaim','ransomnote','hostkidoutcome','hostkidoutcome_txt','weapdetail','nreleased','related','ransomamtus','ransompaid','ransompaidus','ndays','divert','kidhijcountry','ransom','ransomamt','propcomment','ishostkid','nhostkid','nhostkidus','nhours','propextent','propextent_txt','propvalue','nwoundte','nkillus','nwoundus','nkillter','dbsource','INT_LOG','INT_IDEO','INT_MISC','INT_ANY','guncertain3','guncertain2'], axis=1)

df.head()
plt.subplots(figsize=(18,6))

sns.barplot(df['country_txt'].value_counts()[:15].index,df['country_txt'].value_counts()[:15].values,palette='inferno')

plt.title('Top Affected Countries')

plt.show()
sns.barplot(df['targtype1_txt'].value_counts()[1:15].values,df['targtype1_txt'].value_counts()[1:15].index,palette=('inferno'))

plt.xticks(rotation=90)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.title('Favorite Targets')

plt.show()
coun_df=df['iyear'].value_counts()[:15].to_frame()

coun_df.columns=['attacktype1_txt']

coun_kill=df.groupby('iyear')['nkill'].sum().to_frame()

coun_df.merge(coun_kill,left_index=True,right_index=True,how='left').plot.bar(width=0.9)

fig=plt.gcf()

fig.set_size_inches(18,6)

plt.show()
plt.subplots(figsize=(15,6))

sns.countplot('attacktype1_txt',data=df,palette='RdYlGn',order=df['attacktype1_txt'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Attacking Methods by Terrorists')

plt.show()
fig = {

  "data": [

    {

      "values": df['suicide'].value_counts(),

      "labels": df['suicide'].unique(),

        'marker': {'colors': ['rgb(58, 21, 56)',

                                  'rgb(33, 180, 150)']},

      "name": "Gender based suicides",

      "hoverinfo":"label+percent+name",

      "hole": .5,

      "type": "pie"

    }],

     "layout": {

        "title":"The incident was/wasn't a suicide attack"

     }

}



pio.show(fig)
top_groups10=df[df['gname'].isin(df['gname'].value_counts()[1:11].index)]

pd.crosstab(top_groups10.iyear,top_groups10.gname).plot(color=sns.color_palette('Paired',10))

fig=plt.gcf()

fig.set_size_inches(18,6)

plt.show()
from sklearn.cluster import KMeans

from sklearn.metrics.cluster import silhouette_score

from sklearn.preprocessing import scale, robust_scale
df = df[df['nkill'] <= 4].reset_index(drop=True)

df= df[df['nwound'] <= 7].reset_index(drop=True)

c = df.count().sort_values().drop([

    'eventid', 'country', 'iyear', 'natlty1', 'longitude', 'latitude', 'targsubtype1'])

_ = df[c[c > 100000].keys()].var().sort_values().plot.barh()
df.info()
features = [

   

  'longitude',

    'latitude',

    

    'nwound',

    'nkill',

    

    'natlty1_txt',

    'targtype1_txt',

    'targsubtype1_txt',

    'weaptype1_txt',

    'attacktype1_txt',

]



X = pd.get_dummies(df[features])

X = X.T[X.var() > 0.05].T.fillna(0)

X = X.fillna(0)



print('Shape:', X.shape)

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
df['Cluster'] = KMeans(n_clusters=4).fit_predict(X) + 1

names = df.groupby('Cluster')['region_txt'].describe()['top'].values

df['ClusterName'] = df['Cluster'].apply(lambda c: names[c - 1])



numerical = df.dtypes[df.dtypes != 'object'].keys()

exclude = [

    'eventid', 'Cluster', 'region', 'country', 'iyear', 

    'natlty1', 'natlty2', 'natlty3', 'imonth', 'iday',

    'guncertain1'

] + [col for col in numerical if 'type' in col or 'mode' in col or 'ransom' in col]

X_profiling = df[numerical.drop(exclude)].fillna(0)

X_profiling = pd.DataFrame(scale(X_profiling), columns=X_profiling.columns)

X_profiling['ClusterName'] = df['ClusterName']

_ = sns.heatmap(X_profiling.groupby('ClusterName').mean().drop(['longitude', 'latitude'], axis=1).T, 

               cmap='coolwarm')
ckeys = df['ClusterName'].unique()

ckeys = dict(zip(ckeys, plt.cm.tab10(range(len(ckeys)))))



for i, x in X_profiling.groupby('ClusterName'):

    _ = plt.scatter(x['longitude'], x['latitude'], c=ckeys[i], marker='.', cmap='tab10', label=i)

_ = plt.legend(loc=3)