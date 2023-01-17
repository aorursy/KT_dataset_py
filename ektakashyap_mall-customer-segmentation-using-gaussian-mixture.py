# Calling Libraries



# Warnings

import warnings

warnings.filterwarnings("ignore")



# Data manipulation library

import pandas as pd

import numpy as np

import re



# Plotting library

import seaborn as sns; sns.set(style="white", color_codes=True)

import plotly.express as px

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

from pandas.plotting import andrews_curves



# Modeling librray

# Class to develop kmeans model

from sklearn.cluster import KMeans

# Scale data

from sklearn.preprocessing import StandardScaler

# Split dataset

from sklearn.model_selection import train_test_split

from sklearn.mixture import GaussianMixture

from sklearn.manifold import TSNE



# How good is clustering?

from sklearn.metrics import silhouette_score

from yellowbrick.cluster import SilhouetteVisualizer



# os related

import os
# Display multiple outputs from a jupyter cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# Set numpy options to display wide array

np.set_printoptions(precision = 3,          # Display upto 3 decimal places

                    threshold=np.inf        # Display full array

                    )
# Seting display options

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 100)
#os.chdir("E:\HPS\Finance & Accounts\One time activity\SS\Python for analytics\Class Notes & Recordings\Class-13_21-06-20\Excercises & Solutions")

#os.listdir()
# Reading dataset

df = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
df.dtypes
# cleaning/renaming column names

df.rename(columns={'Annual Income (k$)':'AnnualIncome(k$)',

                   'Spending Score (1-100)':'SpendingScore(1-100)',

                  }, 

          inplace=True)                         

df.head()
# Dropping CustomerID column

df.drop(['CustomerID'], inplace = True, axis = 1) 
df.head()
# Transforming Gender

#pd.Series(np.where(df.Gender.values == 'Male', 1, 0, inplace = True))

df.Gender[df.Gender == 'Male'] = 1

df.Gender[df.Gender == 'Female'] = 0

#df.Gender.map(dict(Male=1, Female=0))
df.head()
df["age_cat"] = pd.cut(

                       df['Age'],

                       bins = [0,35,55,100],           # Else devise your bins: [0,20,60,110]

                       labels= ["y", "m", "s"]

                      )
df["income_cat"] = pd.cut(

                           df['AnnualIncome(k$)'],

                           bins = 3,

                           labels= ["l", "m", "h"]

                         )
df.head()
# Shows the relationship of each figure



sns.pairplot(hue="age_cat",data=df)
# Shows the relationship between annual income and spending score by line plot



sns.relplot(x='AnnualIncome(k$)', y='SpendingScore(1-100)', col='age_cat',kind="line",data=df)
sns.relplot(x='Age', y='SpendingScore(1-100)', kind="line",data=df)

sns.relplot(x='Age', y='AnnualIncome(k$)', kind="line",data=df)
#sns.jointplot(x="x", y="y", data=df)

sns.jointplot(x='Age', y='SpendingScore(1-100)', kind = "hex", data=df)

sns.jointplot(x='Age', y='AnnualIncome(k$)', kind="kde",data=df)
sns.catplot(x='age_cat', y='SpendingScore(1-100)', kind = "bar", hue = 'Gender', data=df)

sns.catplot(x='age_cat', y='AnnualIncome(k$)', kind="bar", hue = 'Gender', data=df)
sns.catplot(x='income_cat', y='SpendingScore(1-100)', kind = "bar", hue = 'age_cat', data=df)
sns.catplot(x='income_cat', y='SpendingScore(1-100)', kind = "bar", hue = 'Gender', data=df)
#box plot Gender vs Annual_Income_k

sns.boxplot(x = 'Gender', y = 'AnnualIncome(k$)', data = df)
sns.heatmap(df.corr(), linecolor = 'black', linewidth = 1, annot = True)
#Drop Categorical Values

df.drop(columns=['age_cat', 'income_cat'], inplace=True)
# Scaling using StandardScaler

ss = StandardScaler()

ss.fit(df)

X = ss.transform(df)
# 4.1 Perform clsutering

gm = GaussianMixture(

                     n_components = 2,

                     n_init = 10,

                     max_iter = 100)
# 4.2 Train the algorithm

gm.fit(X)
# 4.3 Where are the clsuter centers

gm.means_
# 4.4 Did algorithm converge?

gm.converged_
# 4.5 How many iterations did it perform?

gm.n_iter_
# 4.6 Clusters labels

gm.predict(X)
# 4.7 Weights of respective gaussians.

gm.weights_
# 4.7.1 What is the frequency of data-points

#       for the three clusters. (np.unique()

#       ouputs a tuple with counts at index 1)

np.unique(gm.predict(X), return_counts = True)[1]/len(X)
# 4.8 GMM is a generative model.

#     Generate a sample from each cluster

#     ToDo: Generate digits using MNIST

gm.sample()
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

    

# Look at the plots



fig = plt.figure()

plt.plot([1,2,3,4,5,6,7,8], aic, marker = "d", label = 'aic')

plt.plot([1,2,3,4,5,6,7,8], bic, marker = "d", label = 'bic')

plt.show()
tsne = TSNE(n_components = 2)

tsne_out = tsne.fit_transform(X)

plt.scatter(tsne_out[:, 0], tsne_out[:, 1],

            marker='x',

            s=30,              # marker size

            linewidths=5,      # linewidth of marker edges

            c=gm.predict(X)   # Colour as per gmm

            )
# 5.0 Plot cluster and cluster centers

#     both from kmeans and from gmm



fig = plt.figure()



# 5.1

plt.scatter(X[:, 0], X[:, 1],

            c=gm.predict(X),

            s=2)

# 5.2

plt.scatter(gm.means_[:, 0], gm.means_[:, 1],

            marker='v',

            s=5,               # marker size

            linewidths=5,      # linewidth of marker edges

            color='red'

            )

plt.show()
densities = gm.score_samples(X)

densities
density_threshold = np.percentile(densities,4)

density_threshold
anomalies = X[densities < density_threshold]

anomalies

anomalies.shape
# 6.1 Show anomalous points

fig = plt.figure()

plt.scatter(X[:, 0], X[:, 1], c = gm.predict(X))

plt.scatter(anomalies[:, 0], anomalies[:, 1],

            marker='v',

            s=5,               # marker size

            linewidths=5,      # linewidth of marker edges

            color='red'

            )

plt.show()
# 7.1 Get first unanomalous data

unanomalies = X[densities >= density_threshold]

unanomalies.shape    # (192, 4)
# 7.2 Transform both anomalous and unanomalous data

#     to pandas DataFrame

df_anomalies = pd.DataFrame(anomalies, columns = df.columns.values)

df_anomalies['UnAn_or_An'] = 'anomalous'   # Create a IIIrd constant column

df_anomalies

df_normal = pd.DataFrame(unanomalies, columns = df.columns.values)

df_normal['UnAn_or_An'] = 'unanomalous'    # Create a IIIrd constant column
# 7.3 Let us see density plots

sns.distplot(df_anomalies['AnnualIncome(k$)'])

sns.distplot(df_normal['AnnualIncome(k$)'])
# 7.3 Let us see density plots

sns.distplot(df_anomalies['SpendingScore(1-100)'])

sns.distplot(df_normal['SpendingScore(1-100)'])
# 7.4 Draw side-by-side boxplots

# 7.4.1 Ist stack two dataframes

df1 = pd.concat([df_anomalies,df_normal])

# 7.4.2 Draw featurewise boxplots

sns.boxplot(x = df1['UnAn_or_An'], y = df1['SpendingScore(1-100)'])
sns.boxplot(x = df1['UnAn_or_An'], y = df1['AnnualIncome(k$)'])