import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/Mall_Customers.csv')
data.info()
data.head()
data.shape
#Count of males & females

plt.title('Count of Males & Females')

sns.set(style="darkgrid")

sns.countplot(x = 'Gender', data = data)
#Exploring Customer Spending Section

plt.title('Customer Spending Score distribution')

sns.distplot(data['Spending Score (1-100)'], color="g", kde = False)
#Exploring Customer Income Section

plt.title('Customer Income distribution')

sns.distplot(data['Annual Income (k$)'], color="b", kde = True)
plt.title('Customer logged Income distribution')

sns.distplot(np.log(data['Annual Income (k$)']), color="b", kde = True)
sqrt_tran = data.apply(lambda x: x['Annual Income (k$)'] ** 0.5, axis=1)

plt.title('Customer Squared Root Transformed Income distribution')

sns.distplot(sqrt_tran, color="b", kde = True)
#Customer age distribution in terms of gender

plt.title('Customer age distribution in terms of gender')

ax = sns.violinplot(x="Gender", y="Age",

                   data=data, palette="Blues", split=True)
#Customer Income distribution in terms of gender

plt.title('Customer income distribution in terms of gender')

ax = sns.violinplot(x="Gender", y="Annual Income (k$)",

                   data=data, palette="GnBu_d", split=True)
#Customer Spending Score distribution in terms of gender

plt.title('Customer spending distribution in terms of gender')

ax = sns.violinplot(x="Gender", y="Spending Score (1-100)",

                   data=data, palette="coolwarm", split=True)
sns.jointplot("Annual Income (k$)", "Spending Score (1-100)", data=data, color="m")
sns.jointplot("Age", "Spending Score (1-100)", data=data, color="b")
sns.jointplot("Age", "Annual Income (k$)", data=data, color="r")
#Regression plot from Kushal

#https://www.kaggle.com/kushal1996/customer-segmentation-k-means-analysis



plt.figure(1 , figsize = (15 , 7))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

        n += 1

        plt.subplot(3 , 3 , n)

        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

        sns.regplot(x = x , y = y , data = data)

        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )

plt.show()
#Dummy encoding the gender variable

one_hot = pd.get_dummies(data['Gender'])

data_2 = data.join(one_hot)

data_2 = data_2.drop(['Gender', 'Male'], axis = 1)

data_2.head()
data_without_id = data_2.drop('CustomerID', axis = 1)

data_without_id.rename(columns={'Female':'Is_Female', 'Spending Score (1-100)': 'Spending_Score',

                               'Annual Income (k$)':'Annual_Income'}, inplace=True)

colormap = plt.cm.inferno

plt.figure(figsize=(8,6))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(data_without_id.corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
from sklearn.cluster import KMeans

from sklearn import metrics

from scipy.spatial.distance import cdist



# k means determine k

def PlotElbow(X):

    distortions = []

    K = range(1,10)

    for k in K:

        kmeanModel = KMeans(n_clusters=k).fit(X)

        kmeanModel.fit(X)

        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])



    # Plot the elbow

    plt.figure(figsize = (10, 8))

    plt.plot(K, distortions, 'bx-')

    plt.xlabel('k')

    plt.ylabel('Distortion')

    plt.title('The Elbow Method showing the optimal k')

    plt.show()
#We started off by clustering age vs spending score

data_age = data_without_id[['Age' , 'Spending_Score']].iloc[: , :].values



PlotElbow(data_age)
#The follow plot is from Kushal: referring to

#https://www.kaggle.com/kushal1996/customer-segmentation-k-means-analysis

#Fit model and plot the graph

algorithm = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(data_age)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_
h = 0.02

x_min, x_max = data_age[:, 0].min() - 1, data_age[:, 0].max() + 1

y_min, y_max = data_age[:, 1].min() - 1, data_age[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (15 , 7) )

plt.clf()

Z = Z.reshape(xx.shape)

plt.imshow(Z , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')



plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = data , c = labels1 , 

            s = 200 )

plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')

plt.show()
#Then we do spending score vs annual income

data_income = data_without_id[['Annual_Income' , 'Spending_Score']].iloc[: , :].values



PlotElbow(data_income)
#As we expected, k = 5 is the right cluster where the elbow forms

#Fit model and plot the graph

algorithm = (KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(data_income)

labels2 = algorithm.labels_

centroids2 = algorithm.cluster_centers_



h = 0.02

x_min, x_max = data_income[:, 0].min() - 1, data_income[:, 0].max() + 1

y_min, y_max = data_income[:, 1].min() - 1, data_income[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])



plt.figure(1 , figsize = (15 , 7) )

plt.clf()

Z = Z.reshape(xx.shape)

plt.imshow(Z , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')



plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = data , c = labels2 , 

            s = 200 )

plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.ylabel('Spending Score (1-100)') , plt.xlabel('Anuual Income')

plt.show()
data_all = data_without_id[['Is_Female','Age', 'Annual_Income' , 'Spending_Score']].iloc[: , :].values



PlotElbow(data_all)
algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(data_all)

labels3 = algorithm.labels_

centroids3 = algorithm.cluster_centers_
import plotly as py

import plotly.graph_objs as go

py.offline.init_notebook_mode(connected = True)
data['label3'] =  labels3

trace1 = go.Scatter3d(

    x= data['Age'],

    y= data['Spending Score (1-100)'],

    z= data['Annual Income (k$)'],

    mode='markers',

     marker=dict(

        color = data['label3'], 

        size= 20,

        line=dict(

            color= data['label3'],

            width= 12

        ),

        opacity=0.8

     )

)

new_data = [trace1]

layout = go.Layout(

    title= 'Clusters',

    scene = dict(

            xaxis = dict(title  = 'Age'),

            yaxis = dict(title  = 'Spending Score'),

            zaxis = dict(title  = 'Annual Income')

        )

)

fig = go.Figure(data=new_data, layout=layout)

py.offline.iplot(fig)