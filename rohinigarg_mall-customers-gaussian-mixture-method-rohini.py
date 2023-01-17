

%reset -f

from sklearn.cluster import KMeans

#For creating elliptical-shaped clusters

from sklearn.datasets import make_blobs

#OS related

import os

#Data manipulation

import pandas as pd

import numpy as np



#for math functions

import math



# Data processing 

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import normalize



#Graphing

import matplotlib.pyplot as plt

import plotly.graph_objects as go 

import plotly.express as px

from matplotlib.colors import LogNorm

import seaborn as sns

#TSNE

from sklearn.manifold import TSNE



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"





from sklearn.mixture import GaussianMixture



import scipy
os.chdir("../input/customer-segmentation-tutorial-in-python")

os.listdir()            # List all files in the folder
dfcust=pd.read_csv('Mall_Customers.csv')

dfcust.head()

print("No of Customers:",dfcust.shape[0])

print("No of Columns:",dfcust.shape[1])
dfcust.columns
#rename column names

new_col_names  = {

                 'Annual Income (k$)' :  'Annual_Income_k$',

                 'Spending Score (1-100)' : 'Spending_Score_1_100',

              }

dfcust.rename(new_col_names,inplace=True,axis=1)

dfcust.columns
#Drop CustomerID..No use of This column

dfcust.drop(columns = ['CustomerID'], inplace = True)

# Add age_cat

dfcust["age_cat"] = pd.cut(

                       dfcust['Age'],

                       bins = [-1,29,55,110],           

                       labels= ["y", "m", "s"]

                      )

#Check null value

dfcust.columns[dfcust.isnull().any()]

#Two columns having Null values

#Check how many Null Values

print("Number of Null Values:\n",dfcust[dfcust.columns[dfcust.isnull().any()]].isnull().sum())


dfSummary=dfcust.describe()

dfSummary

dfSummary=dfSummary.T

dfSummary.plot(kind='bar',figsize = (20,8))
#Correlation coefficient : Annual Income Vs Spending Score

g=sns.jointplot(dfcust['Annual_Income_k$'], dfcust.Spending_Score_1_100, kind = 'reg')

s = scipy.stats.linregress(x = dfcust['Annual_Income_k$'],y = dfcust.Spending_Score_1_100)

g.fig.suptitle("Correlation coefficient : Annual Income Vs Spending Score : " + str(s[2]) )



    

#Observation  correlation coefficient : Not a good linear relation between Annual Income and Spending_Score

sns.jointplot(x='Age', y='Spending_Score_1_100', data=dfcust, kind='reg')  # reg, resid, kde, hex



#

#  Observation:  Youger customers are having more scores. Scores drastically decrease after age 40

#
# Gender wise :Age cat Vs Speding Score 

g=sns.catplot(x = 'age_cat',

            y = 'Spending_Score_1_100', 

              hue='Gender',

            kind = 'bar',

             

              estimator=np.sum,

            data = dfcust)

g.fig.set_size_inches(17,15)

#Spending Score is highest  for Middle Aged Customers

#Seniors has least spending score

#In Case of Middle aand Young: Females has more score than Male , but in Seniors Score is more for males

# Heat Map:Age,Gender Vs Speding Score

plt.figure(figsize=(15,10))

grouped = dfcust.groupby(['Age','Gender'])

df_wq = grouped['Spending_Score_1_100'].sum().unstack()

sns.heatmap(df_wq, cmap = plt.cm.coolwarm)

#Most of the customers 

plt.figure(figsize=(15,4))



sns.distplot(dfcust['Annual_Income_k$'])

plt.title('Distribution of Annual Income', fontsize = 10)

plt.xlabel('Annual Income Range')

plt.ylabel('Count')

#box plot Gender vs Spending Score

plt.figure(figsize=(15,4))

sns.boxplot(x = 'Gender',       

            y = 'Spending_Score_1_100',                 

            data = dfcust,

            

            )

#box plot Gender vs Annual_Income_k

sns.boxplot(x = 'Gender',       

            y = 'Annual_Income_k$',                 

            data = dfcust,

            

            )
#Correlation Map for all features

df_corr=dfcust.corr()

plt.figure(figsize = (15, 9))

sns.heatmap(df_corr, linecolor = 'black', linewidth = 1, annot = True)

plt.title('Correlation of customer data\'s features \n Co Relation >0  means  poistive  co linear realtion \n < 0 means opposite Relation ')

plt.show()
#Drop Age Cat

dfcust.drop(columns=['age_cat'],inplace=True)
#Map Gender to numerical values

dfcust.Gender.unique()

dfcust['Gender'] = dfcust['Gender'].map({

                                    'Male' : '0',

                                    'Female' : '1'

                                   }

                               )



dfcust.head()
ss =  StandardScaler()

cc_ss=ss.fit_transform(dfcust)

df_out= pd.DataFrame(cc_ss, columns = dfcust.columns.values)

df_out
bic = []

aic = []

for i in range(8):

    gm = GaussianMixture(

                     n_components = i+1,

                     n_init = 10,

                    max_iter = 100)

    gm.fit(df_out)

    bic.append(gm.bic(df_out))

    aic.append(gm.aic(df_out))

fig = plt.figure()



plt.plot(range(1, 9), aic,marker="o",label="aic")

plt.plot(range(1, 9), bic,marker="o",label="bic")

plt.legend()

plt.show()
gm = GaussianMixture(

                     n_components = 2,

                     n_init = 10,

                     max_iter = 50)

gm.fit(df_out)

    

tsne = TSNE(n_components = 2,perplexity=40.0)

tsne_out = tsne.fit_transform(df_out)



#draw TSNE

plt.scatter(tsne_out[:, 0], tsne_out[:, 1],

            marker='o',

            s=30,              # marker size

            linewidths=5,      # linewidth of marker edges

            c=gm.predict(df_out)   # Colour as per gm

            )
densities = gm.score_samples(df_out)

density_threshold = np.percentile(densities,4)

anomalies      =     df_out[densities < density_threshold]      # Data of anomalous customers

# Unanomalous data

unanomalous =  df_out[densities >= density_threshold]      # Data of unanomalous customers

df_anomaly     =  pd.DataFrame(anomalies, columns = df_out.columns.values)

df_unanomaly = pd.DataFrame(unanomalous, columns = df_out.columns.values)
#@author : Ashok sir 

#a few changes made by me

def densityplots(df1,df2, label1 = "Anomalous",label2 = "Normal"):

    # df1 and df2 are two dataframes

    # As number of features are 17, we have 20 axes

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15,5))

    ax = axes.flatten()

    fig.tight_layout()

   

    for i,col in enumerate(df1.columns):

        # https://seaborn.pydata.org/generated/seaborn.distplot.html

        # For every i, draw two overlapping density plots in different colors

        sns.distplot(df1[col],

                     ax = ax[i],

                     kde_kws={"color": "k", "lw": 3, "label": label1},   # Density plot features

                     hist_kws={"histtype": "step", "linewidth": 2,"alpha": 1, "color": "g"}) # Histogram features

        sns.distplot(df2[col],

                     ax = ax[i],

                     kde_kws={"color": "red", "lw": 3, "label": label2},

                     hist_kws={"histtype": "step", "linewidth": 2,"alpha": 1, "color": "b"})

densityplots(df_anomaly, df_unanomaly, label2 = "Unanomalous")