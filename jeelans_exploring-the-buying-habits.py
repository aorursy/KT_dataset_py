# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from scipy import stats

from sklearn.feature_selection import RFE

from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans

import seaborn as sns

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



import plotly.plotly as py

import plotly.graph_objs as go





import plotly.figure_factory as ff



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

customer = pd.read_csv('../input/Mall_Customers.csv')

# Any results you write to the current directory are saved as output.
print('Number of rows {}, Number of columns {}'.format(customer.shape[0],customer.shape[1]))
customer.dtypes
customer.isnull().sum()
data_report = customer.describe().drop('CustomerID',axis=1).T

def percentile_90(df):

    data_report['90%'] = np.nan

    for column in ['Age','Annual Income (k$)','Spending Score (1-100)']:

        data_report['90%'][column] = np.percentile(df[column],90)

percentile_90(customer)

without_outlier = customer[customer['Annual Income (k$)'] < 117]

data_report['trimmed_mean'] = np.mean(without_outlier).drop('CustomerID')

data_report['trimmed_std'] = np.std(without_outlier).drop('CustomerID')

def interquartile_range(df):

    data_report['interquartile'] = np.nan

    for column in ['Age','Annual Income (k$)','Spending Score (1-100)']:

        data_report['interquartile'][column] = np.percentile(df[column],75) - np.percentile(df[column],25)

interquartile_range(without_outlier)

data_report['MAD'] = without_outlier.mad()

data_report['variable_type'] = np.nan

data_report['variable_type']['Age'] = 'continous numeric values'

data_report['variable_type']['Annual Income (k$)'] = 'continous numeric values'

data_report['variable_type']['Spending Score (1-100)'] = 'descrete numeric values'

data_report['max_z_score'] = np.nan

for column in ['Age','Annual Income (k$)','Spending Score (1-100)']:

    data_report['max_z_score'][column] = max(np.abs(stats.zscore(customer[column])))

data_report
plt.style.use('tableau-colorblind10')

plt.figure(1 , figsize = (15 , 6))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1 , 3 , n)

    plt.subplots_adjust(hspace =20 , wspace = 0.5)

    sns.distplot(customer[x] , bins = 20)

    plt.title('Distplot of {}'.format(x))

plt.show()
plt.figure(1,figsize=(15,5))

sns.countplot(y='Gender',data=customer)

plt.show()
plt.figure(1 , figsize = (15 , 7))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

        n += 1

        plt.subplot(3 , 3 , n)

        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

        sns.regplot(x = x , y = y , data = customer)

        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )

plt.show()
data = []

for gender in ['Male','Female']:

    data.append(go.Scatter(x = customer[customer['Gender']==gender]['Annual Income (k$)'],y = customer[customer['Gender']==gender]['Age'],mode='markers',name=gender))

layout = go.Layout(

    title='Age vs Annual income w.r.t Gender',

    hovermode='closest',

    xaxis= dict(

    title='Annual Income (k$)',

    ticklen=5,

    zeroline=False,

    gridwidth=2),

    yaxis = dict(

    title = 'Rank',

    ticklen = 5,

    gridwidth = 2

    ),

    showlegend = False

)    

figure = go.Figure(data=data,layout=layout)

iplot(figure)
data = []

for gender in ['Male','Female']:

    data.append(go.Scatter(x = customer[customer['Gender']==gender]['Annual Income (k$)'],y = customer[customer['Gender']==gender]['Spending Score (1-100)'],mode='markers',name=gender))

layout = go.Layout(

    title='Age vs Annual income w.r.t Gender',

    hovermode='closest',

    xaxis= dict(

    title='Annual Income (k$)',

    ticklen=5,

    zeroline=False,

    gridwidth=2),

    yaxis = dict(

    title = 'Spending score ',

    ticklen = 5,

    gridwidth = 2

    ),

    showlegend = False

)    

figure = go.Figure(data=data,layout=layout)

iplot(figure)
plt.style.use('fivethirtyeight')

plt.figure(1,figsize=(15,5))

n = 0

for cols in ['Age','Annual Income (k$)','Spending Score (1-100)']:

    n += 1

    plt.subplot(1,3,n)

    plt.subplots_adjust(hspace=0.5,wspace=0.5)

    sns.boxplot(x=cols,y='Gender',data=customer)

    plt.ylabel('Gender' if n== 1 else '')

    plt.title('Box plot' if n==2 else '')

plt.show()
#violinplot

plt.style.use('fivethirtyeight')

plt.figure(1,figsize=(15,5))

n = 0

for cols in ['Age','Annual Income (k$)','Spending Score (1-100)']:

    n += 1

    plt.subplot(1,3,n)

    plt.subplots_adjust(hspace=0.5,wspace=0.5)

    sns.violinplot(x=cols,y='Gender',data=customer)

    plt.ylabel('Gender' if n== 1 else '')

    plt.title('violin plot' if n==2 else '')

plt.show()
#swarmplot

plt.style.use('fast')

plt.figure(1,figsize=(15,5))

n = 0

for cols in ['Age','Annual Income (k$)','Spending Score (1-100)']:

    n += 1

    plt.subplot(1,3,n)

    plt.subplots_adjust(hspace=0.5,wspace=0.5)

    sns.swarmplot(x=cols,y='Gender',data=customer)

    plt.ylabel('Gender' if n== 1 else '')

    plt.title('swarm plot' if n==2 else '')

plt.show()
labels = ['Male','Female']

sizes = [customer.query('Gender == "Male"').Gender.count(),customer.query('Gender == "Female"').Gender.count()]

#colors

colors = ['#ffdaB9','#66b3ff']

#explsion

explode = (0.05,0.05)

plt.figure(figsize=(8,8)) 

plt.Circle( (0,0), 0.7, color='white')

my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85,explode=explode)

p=plt.gcf()

plt.axis('equal')

p.gca().add_artist(my_circle)

plt.show()
from scipy import stats

max(np.abs(stats.zscore(customer['Age'])))

# last two values have the highest z score
customer['Age group'] = customer['Age']

def age_group(age):

    if age > 50:

        return 'baby boomer'

    elif (age >35) and  (age <= 50) :

        return 'Generation X'

    if (age >= 18) and (age <= 35):

        return 'Millennials'

    if age < 18 :

        return 'iGeneration'

customer['Age group'] = customer['Age group'].apply(age_group)

customer[:15]
sns.countplot(y = 'Age group', data=customer)

plt.show()
customer_male = customer[customer['Gender']=='Male']

customer_female = customer[customer['Gender']=='Female']

customer_male1 = pd.DataFrame(customer_male['Age group'].value_counts()).rename({'Age group':'Male'},axis=1)

customer_female1 = pd.DataFrame(customer_female['Age group'].value_counts()).rename({'Age group':'Female'},axis=1)

customer_gender = customer_male1.join(customer_female1)
trace1 = go.Bar(

    x=['Generation X', 'Millennials', 'baby boomer'],

    y=list(customer_gender['Male']),

    name='Male'

)

trace2 = go.Bar(

    x=['Generation X', 'Millennials', 'baby boomer'],

    y=list(customer_gender['Female']),

    name='Female'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='group'

)



fig = go.Figure(data=data, layout=layout)

iplot(data,filename='bar_chart')
plt.style.use('fivethirtyeight')

plt.figure(1,figsize=(15,5))

n = 0

for cols in ['Annual Income (k$)','Spending Score (1-100)']:

    n += 1

    plt.subplot(1,3,n)

    plt.subplots_adjust(hspace=0.5,wspace=0.5)

    sns.boxplot(x=cols,y='Age group',data=customer)

    plt.ylabel('Age group' if n== 1 else '')

    plt.title('Box plot' if n==2 else '')

plt.show()
plt.style.use('fast')

plt.figure(1,figsize=(15,5))

n = 0

for cols in ['Annual Income (k$)','Spending Score (1-100)']:

    n += 1

    plt.subplot(1,3,n)

    plt.subplots_adjust(hspace=0.5,wspace=0.5)

    sns.swarmplot(x=cols,y='Age group',data=customer)

    plt.ylabel('Age group' if n== 1 else '')

    plt.title('swarm plot' if n==2 else '')

plt.show()
data = []

for age_group in ['Generation X','Millennials','baby boomer']:

    data.append(go.Scatter(x = customer[customer['Age group']==age_group]['Annual Income (k$)'],y = customer[customer['Age group']==age_group]['Spending Score (1-100)'],mode='markers',name=age_group))

layout = go.Layout(

    title='Age vs Annual income w.r.t Age group',

    hovermode='closest',

    xaxis= dict(

    title='Annual Income (k$)',

    ticklen=5,

    zeroline=False,

    gridwidth=2),

    yaxis = dict(

    title = 'Spending score ',

    ticklen = 5,

    gridwidth = 2

    ),

    showlegend = False

)    

figure = go.Figure(data=data,layout=layout)

iplot(figure)
max(customer[customer['Age group']=='Generation X']['Spending Score (1-100)'])

np.mean(customer[customer['Age group']=='Generation X']['Spending Score (1-100)'])

min(customer[customer['Age group']=='Generation X']['Spending Score (1-100)'])
trace1 = {

    "x" : list(customer[customer['Gender']=='Male']['Annual Income (k$)']),

    "y" : list(customer[customer['Gender']=='Male']['Age group']),

    "marker": {"color": "pink", "size": 12}, 

          "mode": "markers", 

          "name": "Male", 

          "type": "scatter"

}

trace2 = {

    "x" : list(customer[customer['Gender']=='Female']['Annual Income (k$)']),

    "y" : list(customer[customer['Gender']=='Female']['Age group']),

    "marker": {"color": "blue", "size": 12}, 

          "mode": "markers", 

          "name": "Female", 

          "type": "scatter"

}

data = [trace1, trace2]

layout = {"title": "Gender Earnings Disparity", 

          "xaxis": {"title": "Annual Income (k$)"} 

          }



fig = go.Figure(data=data, layout=layout)

iplot(fig)
t = np.linspace(-1, 1.2, 2000)

x = customer[customer['Age group'] == 'Generation X']['Annual Income (k$)']

y = customer[customer['Age group'] == 'Generation X']['Spending Score (1-100)']



colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]



fig = ff.create_2d_density(

    x, y, colorscale=colorscale,

    hist_color='rgb(255, 237, 222)', point_size=3

)



iplot(fig, filename='histogram_subplots')
t = np.linspace(-1, 1.2, 2000)

x = customer[customer['Age group'] == 'baby boomer']['Annual Income (k$)']

y = customer[customer['Age group'] == 'baby boomer']['Spending Score (1-100)']



colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]



fig = ff.create_2d_density(

    x, y, colorscale=colorscale,

    hist_color='rgb(255, 237, 222)', point_size=3

)



iplot(fig, filename='histogram_subplots')
t = np.linspace(-1, 1.2, 2000)

x = customer[customer['Age group'] == 'Millennials']['Annual Income (k$)']

y = customer[customer['Age group'] == 'Millennials']['Spending Score (1-100)']



colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]



fig = ff.create_2d_density(

    x, y, colorscale=colorscale,

    hist_color='rgb(255, 237, 222)', point_size=3

)



iplot(fig, filename='histogram_subplots')
group_report = pd.DataFrame(columns = list(np.unique(customer['Age group'])))

group_report = group_report.T

Generation_X = customer[customer['Age group']=='Generation X']

baby_boomer= customer[customer['Age group']=='baby boomer']

Millennials = customer[customer['Age group']=='Millennials']

group_report['average spending score'] = np.nan

group_report['max spending score'] = np.nan

group_report['min spending score'] = np.nan

group_report['average spending score']['Generation X'] = np.mean(Generation_X['Spending Score (1-100)'] )

group_report['average spending score']['Millennials'] = np.mean(Millennials['Spending Score (1-100)'] )

group_report['average spending score']['baby boomer'] = np.mean(baby_boomer['Spending Score (1-100)'] )

group_report['max spending score']['Generation X'] = max(Generation_X['Spending Score (1-100)'] )

group_report['max spending score']['Millennials'] = max(Millennials['Spending Score (1-100)'] )

group_report['max spending score']['baby boomer'] = max(baby_boomer['Spending Score (1-100)'] )

group_report['min spending score']['Generation X'] = min(Generation_X['Spending Score (1-100)'] )

group_report['min spending score']['Millennials'] = min(Millennials['Spending Score (1-100)'] )

group_report['min spending score']['baby boomer'] = min(baby_boomer['Spending Score (1-100)'] )

group_report['average annual income'] = np.nan

group_report['max annual income'] = np.nan

group_report['min annual income'] = np.nan

group_report['average annual income']['Generation X'] = np.mean(Generation_X['Annual Income (k$)'] )

group_report['average annual income']['Millennials'] = np.mean(Millennials['Annual Income (k$)'] )

group_report['average annual income']['baby boomer'] = np.mean(baby_boomer['Annual Income (k$)'] )

group_report['max annual income']['Generation X'] = max(Generation_X['Annual Income (k$)'] )

group_report['max annual income']['Millennials'] = max(Millennials['Annual Income (k$)'] )

group_report['max annual income']['baby boomer'] = max(baby_boomer['Annual Income (k$)'] )

group_report['min annual income']['Generation X'] = min(Generation_X['Annual Income (k$)'] )

group_report['min annual income']['Millennials'] = min(Millennials['Annual Income (k$)'] )

group_report['min annual income']['baby boomer'] = min(baby_boomer['Annual Income (k$)'] )

group_report
data = (

  {"label": "Generation X(min average  max Spending score)", 

   "range": [0,100], "performance": [1,41.7,95]},

  {"label": "Millennials(min average max Spending score)", 

   "range": [0,100], "performance": [1,60,99]},

  {"label": "baby boomer(min average max Spending score)",

   "range": [0,100],"performance": [3,37,60]}

  

)



fig = ff.create_bullet(

    data, titles='label', 

    measures='performance', ranges='range', orientation='v',

)

iplot(fig, filename='bullet chart from dict')
import matplotlib.pyplot as plt

plt.scatter(customer['Spending Score (1-100)'],customer['Annual Income (k$)'],c='blue',marker='o',s=50)

plt.grid()

plt.show()
from sklearn.cluster import KMeans

inertia_list=[]

X= customer[['Spending Score (1-100)','Annual Income (k$)']].values

#we always assume the max number of cluster would be 10

#you can judge the number of clusters by doing averaging

###Static code to get max no of clusters



for i in range(1,11):

    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)

    kmeans.fit(X)

    inertia_list.append(kmeans.inertia_)
plt.plot(range(1,11), inertia_list)

plt.title('The Elbow Method')

plt.xlabel('no of clusters')

plt.ylabel('inertia')

plt.show()
X1= customer[['Spending Score (1-100)','Annual Income (k$)']]

from sklearn.cluster import KMeans

def doKmeans(X, nclust=2):

    model = KMeans(nclust,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)

    model.fit(X)

    clust_labels = model.predict(X)

    cent = model.cluster_centers_

    return (clust_labels, cent)



clust_labels, cent = doKmeans(X1, 5)

kmeans = pd.DataFrame(clust_labels)

X1.insert((X1.shape[1]),'kmeans',kmeans)

X1[:15]
fig = plt.figure()

ax = fig.add_subplot(111)

scatter = ax.scatter(X1['Spending Score (1-100)'],X1['Annual Income (k$)'],

                     c=kmeans[0],s=50)

ax.set_title('K-Means Clustering')

ax.set_xlabel('Spening score (1-100)')

ax.set_ylabel('Annual Income (k$)')

plt.colorbar(scatter)

plt.show()
km = KMeans(n_clusters=5)

y_km = km.fit_predict(X)

import numpy as np

from matplotlib import cm

from sklearn.metrics import silhouette_samples

cluster_labels = np.unique(y_km)

n_clusters = cluster_labels.shape[0]

silhouette_vals = silhouette_samples(X,

y_km,

metric='euclidean')

y_ax_lower, y_ax_upper = 0, 0

yticks = []

for i, c in enumerate(cluster_labels):

     c_silhouette_vals = silhouette_vals[y_km == c]

     c_silhouette_vals.sort()

     y_ax_upper += len(c_silhouette_vals)

     color = cm.jet(i / n_clusters)

     plt.barh(range(y_ax_lower, y_ax_upper),c_silhouette_vals,height=1.0,edgecolor='none',color=color)

     yticks.append((y_ax_lower + y_ax_upper) / 2)

     y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)

plt.axvline(silhouette_avg,

color="red",

linestyle="--")

plt.yticks(yticks, cluster_labels + 1)

plt.show()