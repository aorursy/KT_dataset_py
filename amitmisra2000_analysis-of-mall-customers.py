# 1.0 Call libraries

from sklearn.cluster import KMeans

# 1.1 For creating elliptical-shaped clusters

from sklearn.datasets import make_blobs

# 1.2 Data manipulation

import pandas as pd

import numpy as np

# 1.3 Plotting

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

import seaborn as sns



#1.4Import SKLearn Classes

from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture

from sklearn.manifold import TSNE

import re  #regular expression

from sklearn.preprocessing import StandardScaler

from pandas.plotting import andrews_curves



import warnings

warnings.filterwarnings('ignore')

# 1.5 TSNE

from sklearn.manifold import TSNE

import os

# 1.6 Set your working folder to where data is

os.chdir("../input")

os.listdir()



#1.7 Import Warnings

import warnings

warnings.filterwarnings('ignore')

# 2.1 Read csv file

Mall_Customers = pd.read_csv("Mall_Customers.csv")

Mall_Customers_col = {

'Annual Income (k$)':'Annual_Income',

'Spending Score (1-100)':'Spending_Score'

}



Mall_Customers.rename(columns = Mall_Customers_col,inplace = True)



#2.2 Drop Customer ID Column from Mall Customer Table

Mall_Customers.drop('CustomerID',axis =1,inplace = True)



#2.3 Replacing {Male,Female] in gender column with [0,1]



Mall_Customers.Gender.replace(('Male', 'Female'), (1, 0), inplace=True)





# Engineering  features

Mall_Customers["Spending_Score_Cat"] = pd.cut(

                       Mall_Customers['Spending_Score'],

                       bins = [0,33,66,99],

                       labels= [1,2,3]

                      )



Mall_Customers['Spending_Score_Cat'] =Mall_Customers['Spending_Score_Cat'].astype(str).astype(int)

Mall_Customers.dtypes

#2.4 Plotting of Mall Customer Data



sns.distplot(Mall_Customers.Age)  #Mall Age Data is More Scewed towards Young Age.

sns.despine()     





sns.distplot(Mall_Customers.Annual_Income)



#2.5 Columns in num_data that are either discrete (with few levels)

#     or numeric

num_data = Mall_Customers.select_dtypes(include = ['float64', 'int64','int32']).copy()

cols= ['Age','Gender','Annual_Income','Spending_Score_Cat']
# 3.0 Create an instance of StandardScaler object

ss= StandardScaler()



# 3.1 Use fit and transform method

nc = ss.fit_transform(num_data.loc[:,cols])



# 3.3

nc.shape     # (200,4)



# 3.4 Transform numpy array back to pandas dataframe

#        as we will be using pandas plotting functions

nc = pd.DataFrame(nc, columns = cols)



# 3.6 Add/overwrite few columns that are discrete

#        These columns were not to be scaled



nc['Gender'] = Mall_Customers['Gender']

# 3.7 Also create a dataframe from random data

#      for comparison

nc_rand = pd.DataFrame(np.random.randn(200,4),

                       columns = cols    # Assign column names, just like that

                       )



# 3.8 Add/overwrite these columns also

nc_rand['Spending_Score_Cat']        = np.random.randint(3, size= (200,))   # [0,1]

nc_rand.shape    # (200,4)

#4.0 Now start plotting



#4.1 Parallel coordinates with random data

fig1 = plt.figure()

pd.plotting.parallel_coordinates(nc_rand,

                                 'Spending_Score_Cat',    # class_column

                                  colormap='winter'

                                  )

plt.xticks(rotation=90)

plt.title("Parallel chart with random data")





# 4.2 Parallel coordinates with 'Mall Customer' data

fig2 = plt.figure()

ax = pd.plotting.parallel_coordinates(nc,'Spending_Score_Cat',

                                  colormap= plt.cm.winter

                                  )



plt.xticks(rotation=90)

plt.title("Parallel chart with Mall Customer data")



# 4.3 Andrews charts with random data

fig3 = plt.figure()

pd.plotting.andrews_curves(nc_rand,

                           'Spending_Score_Cat',

                           colormap = 'winter')



plt.title("Andrews plots with random data")





# 4.4 Andrews plots with Mall Customer data

fig4 = plt.figure()

pd.plotting.andrews_curves(nc,

                           'Spending_Score_Cat',

                            colormap =  plt.cm.winter

                           )

plt.xticks(rotation=90)

plt.title("Andrews curve with Mall_Customer data")



#5.1 Done Unsupervised Learning Using KMeans Algorithm

ss= StandardScaler()

ss.fit(Mall_Customers)

X = ss.transform(Mall_Customers)

X.shape



sse = []

no_of_clusters = 30

for k in range(1,no_of_clusters):

    km = KMeans(n_clusters = k)

    km.fit(X)

    sse.append(km.inertia_)

plt.plot(range(1,no_of_clusters), sse, marker='*')





#5.2 This algorithm shows 7 Clusters are optimum now we see what is interpretion of all the 7 clusters



kmean= KMeans(7)

kmean.fit(X)

labels=kmean.labels_



clusters=pd.concat([Mall_Customers, pd.DataFrame({'cluster':labels})], axis=1)

clusters.head()

for c in clusters:

    grid= sns.FacetGrid(clusters, col='cluster')

    grid.map(plt.hist, c)
#6.1 Import GaussianMixture class

from sklearn.mixture import GaussianMixture

ss= StandardScaler()

ss.fit(Mall_Customers)

X = ss.transform(Mall_Customers)

X.shape



#6.2 Done Unsuperwised learning using Guassian Mixture Model

gm = GaussianMixture(

                     n_components = 5,

                     n_init = 10,

                     max_iter = 100)

gm.fit(X)



fig = plt.figure()



plt.scatter(X[:, 1], X[:, 2],

            c=gm.predict(X),

            s=5)

plt.scatter(gm.means_[:, 1], gm.means_[:, 2],

            marker='v',

            s=10,               # marker size

            linewidths=5,      # linewidth of marker edges

            color='red'

            )
#7.1 Findings of BIC and AIC

bic =[]

aic= []



for i in range(8):

    gm = GaussianMixture(

                     n_components = i+1,

                     n_init = 10,

                     max_iter = 100)

    gm.fit(X)

    bic.append(gm.bic(X))

    aic.append(gm.aic(X))

    

fig = plt.figure()

plt.plot([1,2,3,4,5,6,7,8], aic)

plt.plot([1,2,3,4,5,6,7,8], bic)

plt.show()
#8.1Now do unsupervised learning using TSNE Model



tsne = TSNE(n_components = 2)

tsne_out = tsne.fit_transform(X)

plt.scatter(tsne_out[:, 0], tsne_out[:, 1],

            marker='x',

            s=50,              # marker size

            linewidths=5,      # linewidth of marker edges

            c=gm.predict(X)   # Colour as per gmm

            )
