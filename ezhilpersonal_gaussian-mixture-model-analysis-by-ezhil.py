%reset -f



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture

from sklearn.manifold import TSNE

import re  #regular expression

from sklearn.preprocessing import StandardScaler

from pandas.plotting import andrews_curves

from mpl_toolkits.mplot3d import Axes3D
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
pd.options.display.max_rows = 1000

pd.options.display.max_columns = 1000
%matplotlib inline
customer = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

customer.shape

customer.head()
new_columns = {x : re.sub('[^A-Za-z]+','',x) for x in customer.columns.values}

new_columns

customer.rename(columns = new_columns,inplace=True)

customer.rename(columns = {"AnnualIncomek": "AnnualIncome"},inplace=True)
customer["Gender"].value_counts()

customer["GenderCode"] = customer["Gender"].map({"Female" : 0, "Male" : 1})
customer.drop(columns=["CustomerID","Gender"], inplace=True)
customer.head()
customer.info()
customer.describe()
values = customer["GenderCode"].value_counts()

ax = sns.countplot(customer["GenderCode"])

for i, p in enumerate(ax.patches):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2., height + 0.1, values[i],ha="center")
andrews_curves(customer, "GenderCode")
fig = plt.figure(figsize=(15,5))

ax=plt.subplot(1,3,1)

sns.boxplot(data=customer, x="GenderCode",y="Age")

ax=plt.subplot(1,3,2)

sns.boxplot(data=customer, x="GenderCode",y="AnnualIncome")

ax=plt.subplot(1,3,3)

sns.boxplot(data=customer, x="GenderCode",y="SpendingScore")

fig = plt.figure(figsize=(15,5))

ax=plt.subplot(1,3,1)

sns.stripplot(data=customer, x="GenderCode",y="Age")

ax=plt.subplot(1,3,2)

sns.stripplot(data=customer, x="GenderCode",y="AnnualIncome")

ax=plt.subplot(1,3,3)

sns.stripplot(data=customer, x="GenderCode",y="SpendingScore")
fig = plt.figure(figsize=(15,5))

ax=plt.subplot(1,3,1)

sns.swarmplot(data=customer, x="GenderCode",y="Age")

ax=plt.subplot(1,3,2)

sns.swarmplot(data=customer, x="GenderCode",y="AnnualIncome")

ax=plt.subplot(1,3,3)

sns.swarmplot(data=customer, x="GenderCode",y="SpendingScore")
fig = plt.figure(figsize=(15,5))

ax=plt.subplot(1,3,1)

sns.distplot(customer.Age, rug=True)

ax=plt.subplot(1,3,2)

sns.distplot(customer.AnnualIncome, rug=True)

ax=plt.subplot(1,3,3)

sns.distplot(customer.SpendingScore, rug=True)
sns.pairplot(customer, vars=["Age","AnnualIncome","SpendingScore"], diag_kind="kde"

             , kind="reg", hue="GenderCode", markers=["o","D"],palette="husl")
fig = plt.figure(figsize=(15,5))

ax=plt.subplot(1,3,1)

sns.scatterplot(data=customer, x="Age",y="AnnualIncome")

ax=plt.subplot(1,3,2)

sns.scatterplot(data=customer, x="Age",y="SpendingScore")

ax=plt.subplot(1,3,3)

sns.scatterplot(data=customer, x="AnnualIncome",y="SpendingScore")
fig = plt.figure(figsize=(15,5))

ax=plt.subplot(1,3,1)

sns.regplot(data=customer, x="Age",y="AnnualIncome")

ax=plt.subplot(1,3,2)

sns.regplot(data=customer, x="Age",y="SpendingScore")

ax=plt.subplot(1,3,3)

sns.regplot(data=customer, x="AnnualIncome",y="SpendingScore")
px.scatter(customer.sort_values(by="Age"),

          x = "AnnualIncome",

          y = "SpendingScore",

          #size = "GenderCode",

          range_x=[0,140],

          range_y=[0,100] ,

          animation_frame = "Age", 

          animation_group = "GenderCode", 

          color = "GenderCode" 

          )
fig = plt.figure(figsize=(10,5))

ax = plt.axes(projection='3d')

ax.scatter3D(customer['Age'], customer['AnnualIncome'], customer['SpendingScore']

             , c=customer['GenderCode'], cmap='RdBu');

ax.set_xlabel('Age')

ax.set_ylabel('AnnualIncome')

ax.set_zlabel('SpendingScore')
fig = plt.figure(figsize=(10,5))

ax = plt.axes(projection='3d')

ax.plot(customer['Age'], customer['AnnualIncome'], customer['SpendingScore']);

ax.set_xlabel('Age')

ax.set_ylabel('AnnualIncome')

ax.set_zlabel('SpendingScore')
ss= StandardScaler()

ss.fit(customer)

X = ss.transform(customer)

X.shape
sse = []

for k in range(1,10):

    km = KMeans(n_clusters = k)

    km.fit(X)

    sse.append(km.inertia_)

plt.plot(range(1,10), sse, marker='*')
bic = []

aic = []

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
kmeans_bad = KMeans(n_clusters=2,

                    n_init =10,

                    max_iter = 800)

kmeans_bad.fit(X)



centroids=kmeans_bad.cluster_centers_



fig = plt.figure()

plt.scatter(X[:, 1], X[:, 2],

            c=kmeans_bad.labels_,

            s=2)

plt.scatter(centroids[:, 1], centroids[:, 2],

            marker='x',

            s=100,               # marker size

            linewidths=150,      # linewidth of marker edges

            color='red'

            )

plt.show()
gm = GaussianMixture(

                     n_components = 2,

                     n_init = 10,

                     max_iter = 100)

gm.fit(X)

#gm.means_

#gm.converged_

#gm.n_iter_

#gm.predict(X)

#gm.weights_

#np.unique(gm.predict(X), return_counts = True)[1]/len(X)

#gm.sample()

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

plt.show()

gm = GaussianMixture(

                     n_components = 2,

                     n_init = 10,

                     max_iter = 100)

gm.fit(X)



tsne = TSNE(n_components = 2)

tsne_out = tsne.fit_transform(X)

plt.scatter(tsne_out[:, 0], tsne_out[:, 1],

            marker='x',

            s=50,              # marker size

            linewidths=5,      # linewidth of marker edges

            c=gm.predict(X)   # Colour as per gmm

            )
densities = gm.score_samples(X)

densities



density_threshold = np.percentile(densities,5)

density_threshold



anomalies = X[densities < density_threshold]

anomalies

anomalies.shape







fig = plt.figure()

plt.scatter(X[:, 1], X[:, 2], c = gm.predict(X))

plt.scatter(anomalies[:, 0], anomalies[:, 1],

            marker='x',

            s=50,               # marker size

            linewidths=5,      # linewidth of marker edges

            color='red'

            )

unanomalies = X[densities >= density_threshold]

unanomalies.shape   



df_anomalies = pd.DataFrame(anomalies[:,[1,2]], columns=['salary','spendingscore'])

df_anomalies['type'] = 'anomalous'   # Create a IIIrd constant column

df_normal = pd.DataFrame(unanomalies[:,[1,2]], columns=['salary','spendingscore'])

df_normal['type'] = 'unanomalous'    # Create a IIIrd constant column



df_anomalies.head()

df_normal.head()


# 7.3 Let us see density plots

sns.distplot(df_anomalies['salary'], color='orange')

sns.distplot(df_normal['salary'], color='blue')

sns.distplot(df_anomalies['spendingscore'], color='orange')

sns.distplot(df_normal['spendingscore'], color='blue')


df = pd.concat([df_anomalies,df_normal])

df_anomalies.shape

df_normal.shape

df.shape

sns.boxplot(x = df['type'], y = df['salary'])
sns.boxplot(x = df['type'], y = df['spendingscore'])
customer_NoGender = customer.copy() #Deep Copy

customer_NoGender.drop(columns=["GenderCode"], inplace = True)

#customer.head()

customer_NoGender.head()
ss= StandardScaler()

ss.fit(customer_NoGender)

X = ss.transform(customer_NoGender)
bic = []

aic = []

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
gm = GaussianMixture(

                     n_components = 5,

                     n_init = 10,

                     max_iter = 100)

gm.fit(X)

#gm.means_

#gm.converged_

#gm.n_iter_

#gm.predict(X)

#gm.weights_

#np.unique(gm.predict(X), return_counts = True)[1]/len(X)

#gm.sample()

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
gm = GaussianMixture(

                     n_components = 5,

                     n_init = 10,

                     max_iter = 100)

gm.fit(X)
tsne = TSNE(n_components = 2)

tsne_out = tsne.fit_transform(X)

plt.scatter(tsne_out[:, 0], tsne_out[:, 1],

            marker='x',

            s=50,              # marker size

            linewidths=5,      # linewidth of marker edges

            c=gm.predict(X)   # Colour as per gmm

            )