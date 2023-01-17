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

# 1.4 TSNE

from sklearn.manifold import TSNE

import os

import matplotlib.pyplot as plt

import plotly.express as px

import matplotlib

from sklearn.preprocessing import StandardScaler

%matplotlib inline

# Display output of command in a cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# Go to folder containing data file

#os.chdir("E:\\IDA_Training\\Assignments\\GMM")

#os.listdir()            # List all files in the folder



#df = pd.read_csv("datasets_42674_74935_Mall_Customers.csv")

os.chdir("/kaggle/input/customer-segmentation-tutorial-in-python")

df = pd.read_csv("Mall_Customers.csv")

df.head()
df.shape               

df.dtypes
df.columns = df.columns.str.replace("k", "")

df.columns = df.columns.str.replace("$", "")

df.columns = df.columns.str.replace("1", "")

df.columns = df.columns.str.replace("0", "")

df.columns = df.columns.str.replace("-", "")

df.columns = df.columns.str.replace("(", "")

df.columns = df.columns.str.replace(")", "")

df.columns = df.columns.str.strip()

df.columns = df.columns.str.replace(" ", "_")

df.head()
df['Gender'] = df['Gender'].map({

                                    'Male' : 1,

                                    'Female' : 0

                               })

df.drop(['CustomerID'], axis=1,inplace = True)
df.head()
sns.distplot(df.Age)
columns = ['Age', 'Annual_Income', 'Spending_Score']

fig = plt.figure(figsize = (15,15))

for i in range(len(columns)):

    plt.subplot(2,2,i+1)

    sns.distplot(df[columns[i]])
columns = ['Age', 'Annual_Income', 'Spending_Score']

catVar = ['Gender' ]





# 6.3 Now for loop. First create pairs of cont and cat variables

mylist = [(cont,cat)  for cont in columns  for cat in catVar]

mylist



# 6.4 Now run-through for-loop

fig = plt.figure(figsize = (20,20))

for i, k in enumerate(mylist):

    #print(i, k[0], k[1])

    plt.subplot(4,2,i+1)

    sns.boxplot(x = k[1], y = k[0], data = df,notch = True)
from mpl_toolkits.mplot3d import Axes3D

#columns = ['Age', 'Annual_Income', 'Spending_Score']

fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111, projection='3d')



xs = df['Age']

ys = df['Annual_Income']

zs = df['Spending_Score']

ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')



ax.set_xlabel('Age')

ax.set_ylabel('Annual_Income')

ax.set_zlabel('Spending_Score')


from sklearn import preprocessing

from sklearn.decomposition import PCA



ss = StandardScaler()     # Create an instance of class

ss.fit(df)                # Train object on the data

X = ss.transform(df)      # Transform data

X[:2, :]

normalized_df = preprocessing.normalize(X)

normalized_df = pd.DataFrame(normalized_df)

pca = PCA(n_components = 2) 

X_principal = pca.fit_transform(normalized_df) 

X_principal = pd.DataFrame(X_principal) 

X_principal.columns = ['P1', 'P2'] 

X_principal.head(2)
from sklearn.mixture import GaussianMixture



#gmm = GaussianMixture(n_components = 3) 

#gmm.fit(X_principal)



# 4.1 Perform clsutering

gm = GaussianMixture(

                     n_components = 3,

                     n_init = 10,

                     max_iter = 100)



# 4.2 Train the algorithm

gm.fit(X_principal)



# 4.3 Where are the clsuter centers

gm.means_



# 4.4 Did algorithm converge?

gm.converged_



# 4.5 How many iterations did it perform?

gm.n_iter_



gm.predict(X_principal)
plt.scatter(X_principal['P1'], X_principal['P2'], 

            c=gm.predict(X_principal),

            s=2)

plt.scatter(gm.means_[:, 0], gm.means_[:, 1],

            marker='v',

            s=5,               # marker size

            linewidths=5,      # linewidth of marker edges

            color='red'

            )

            

#           c = GaussianMixture(n_components = 3).fit_predict(X_principal), cmap =plt.cm.winter, alpha = 0.6) 

plt.show()
densities = gm.score_samples(X_principal)

#densities



density_threshold = np.percentile(densities,5)

density_threshold



anomalies = X_principal[densities < density_threshold]



anomalies.shape
fig = plt.figure()

plt.scatter(X_principal['P1'], X_principal['P2'], 

            c=gm.predict(X_principal),

            s=2)

#plt.scatter(X_principal[:, 0], X_principal[:, 1], c = gm.predict(X_principal))



plt.scatter(anomalies['P1'], anomalies['P2'],

            marker='x',

            s=20,               # marker size

            linewidths=100,      # linewidth of marker edges

            color='red'

            )

plt.show()
bic = []

aic = []

for i in range(8):

    gm = GaussianMixture(

                     n_components = i+1,

                     n_init = 10,

                     max_iter = 100)

    gm.fit(X_principal)

    bic.append(gm.bic(X_principal))

    aic.append(gm.aic(X_principal))
fig = plt.figure()

plt.plot([1,2,3,4,5,6,7,8], aic)

plt.plot([1,2,3,4,5,6,7,8], bic)

plt.show()
unanomalies = X_principal[densities >= density_threshold]

unanomalies.shape    # (1200, 2)

# 7.2 Transform both anomalous and unanomalous data

#     to pandas DataFrame

df_anomalies = pd.DataFrame(anomalies, columns = ['P1', 'P2'])

df_anomalies['z'] = 'anomalous'   # Create a IIIrd constant column

df_normal = pd.DataFrame(unanomalies, columns = ['P1','P2'])

df_normal['z'] = 'unanomalous'    # Create a IIIrd constant column

df = pd.concat([df_anomalies,df_normal])

df.head()



# 7.4.2 Draw featurewise boxplots

sns.boxplot(x = df['z'], y = df['P2'])

#sns.boxplot(x = df['z'], y = df['P1'])

plt.show()

sns.boxplot(x = df['z'], y = df['P1'])
sns.distplot(df_anomalies['P1'])

sns.distplot(df_normal['P1'])