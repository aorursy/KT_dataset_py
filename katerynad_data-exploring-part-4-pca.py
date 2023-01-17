import pandas as pd

from pandas import Series,DataFrame

import numpy as np

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler





# For Visualization

import matplotlib.pyplot as plt

import matplotlib

from math import sqrt

import seaborn as sns

#3D

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.patches import FancyArrowPatch

from mpl_toolkits.mplot3d import proj3d

#data

df=pd.read_csv('../input/indicators_by_company.csv')

df.head(5)
indicators=['Assets','LiabilitiesAndStockholdersEquity',

'StockholdersEquity',

'CashAndCashEquivalentsAtCarryingValue',

'NetCashProvidedByUsedInOperatingActivities',

'NetIncomeLoss',

'NetCashProvidedByUsedInFinancingActivities',

'CommonStockSharesAuthorized',

'CashAndCashEquivalentsPeriodIncreaseDecrease',

'CommonStockValue',

'CommonStockSharesIssued',

'RetainedEarningsAccumulatedDeficit',

'CommonStockParOrStatedValuePerShare',

'NetCashProvidedByUsedInInvestingActivities',

'PropertyPlantAndEquipmentNet',

'AssetsCurrent',

'LiabilitiesCurrent',

'CommonStockSharesOutstanding',

'Liabilities',

'OperatingIncomeLoss' ]
Values=df.loc[df['indicator_id'].isin(indicators),['company_id','indicator_id','2011']]

Values=pd.melt(Values, id_vars=['company_id', 'indicator_id'], var_name='year', value_name='value')

Values=Values.loc[Values['year']=='2011',['company_id','indicator_id','value']].pivot(index='company_id',columns='indicator_id', values='value').dropna()

Values.head(5)
scaler = StandardScaler().fit(Values[indicators])

Values_Scaled = scaler.transform(Values[indicators])



print(Values_Scaled[:,0].mean())  

print(Values_Scaled[:,0].std())  
var_exp=[]

cum_var_exp=[]

pca = PCA(n_components=10)

pca.fit(Values_Scaled)

var_exp=pca.explained_variance_ratio_

cum_var_exp = np.cumsum(var_exp)
with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(10, 8))



    plt.bar(range(10), var_exp, alpha=0.5, align='center',

            label='individual explained variance')

    plt.step(range(10), cum_var_exp, where='mid',

             label='cumulative explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
pca = PCA(n_components=6)

pc_scores = pd.DataFrame(pca.fit_transform(Values_Scaled))

pc_scores.columns = ['PC'+str(i+1) for i in range(len(pc_scores.columns))]

pc_scores.head()
#principal components

pc = pd.DataFrame(pca.components_, columns=indicators)

pc.index = ['PC'+str(i+1) for i in range(len(pc_scores.columns))]

pc.head()
round(pca.explained_variance_ratio_.sum()*100)
#heatmap visualization

def heatmap(data):

  fig, ax = plt.subplots(figsize=(10, 10))

  heatmap = sns.heatmap(data, cmap=plt.cm.Blues, center=0, linewidths=0.5, 

                  vmin=-1, vmax=1,annot=True, annot_kws={"size": 8})

  ax.xaxis.tick_top()  

# rotate

  plt.xticks(rotation=90)

  plt.yticks(rotation=0)

  plt.tight_layout()

 # Biplots

# Thanks to  DR-Rodriguez

# https://www.kaggle.com/strakul5/d/abcsds/pokemon/principal-component-analysis-of-pokemon-data

def pca_biplot(x_pc=0, y_pc=1, max_arrow=0.2):

    n = pc.shape[1]

    sns.set(style="ticks", palette="muted", color_codes=True)

    

    g = sns.lmplot(x='PC{}'.format(x_pc + 1), y='PC{}'.format(y_pc + 1),  data=pc_scores,

                   fit_reg=False, size=8)

    for i in range(n):

        # Only plot the longer ones

        length = sqrt(pc.iloc[x_pc, i] ** 2 + pc.iloc[y_pc, i] ** 2)

        if length < max_arrow:

            continue

        plt.arrow(0, 0, pc.iloc[x_pc, i], pc.iloc[y_pc, i], color='k', alpha=0.9)

        plt.text(pc.iloc[x_pc, i] * 1.15, pc.iloc[y_pc, i] * 1.15,

                 pc.columns.tolist()[i], color='k', ha='center', va='center')

    g.set(ylim=(-1, 1))

    g.set(xlim=(-1, 1))



class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):

         FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)

         self._verts3d = xs, ys, zs

    def draw(self, renderer):

        xs3d, ys3d, zs3d = self._verts3d

        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)

        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        FancyArrowPatch.draw(self, renderer)

def pca_3Dplot(x_pc=0, y_pc=1, z_pc=2, max_arrow=0.2):        

    fig = plt.figure(1, figsize=(8, 6))

    ax = Axes3D(fig, elev=-150, azim=50)

    sns.set(style="ticks", palette="muted", color_codes=True)



    ax.scatter(pc_scores.iloc[:, x_pc], pc_scores.iloc[:, y_pc], pc_scores.iloc[:, z_pc], 

           cmap='plt.cm.Paired')

    n = pc.shape[1]

    for i in range(n):

        length = sqrt(pc.iloc[0, i] ** 2 + pc.iloc[1, i] ** 2+pc.iloc[2, i] ** 2)

        if length < max_arrow:

            continue

        a = Arrow3D([0, pc.iloc[0, i]], [0, pc.iloc[1, i]], 

                [0, pc.iloc[2, i]], mutation_scale=20, 

                lw=2, arrowstyle="-|>", color="r")

        ax.add_artist(a)

        ax.text(x=pc.iloc[x_pc, i]*1.15, y=pc.iloc[y_pc, i]*1.15, z=pc.iloc[z_pc, i]*1.15,

                 s=pc.columns.tolist()[i],color='k', ha='center', va='center')

    ax.set_title("Three PCA directions")

    ax.set_xlabel('PC{}'.format(x_pc + 1))

    ax.w_xaxis.set_ticklabels([])

    ax.set_ylabel('PC{}'.format(y_pc + 1))

    ax.w_yaxis.set_ticklabels([])

    ax.set_zlabel('PC{}'.format(z_pc + 1))

    ax.w_zaxis.set_ticklabels([])

    ax.set_xlim3d(-1, 1)

    ax.set_ylim3d(-1, 1)

    ax.set_zlim3d(-1, 1)
heatmap(pc.transpose()**2)
pca_biplot( 0, 1, max_arrow=0.5)
heatmap(pc.transpose())
# PC3 vs PC4

pca_biplot( 2, 3, max_arrow=0.5)
#PC1, PC2 and PC3

pca_3Dplot(0,1,2,0.5)
#PC2, PC3, PC4

pca_3Dplot(1,2,3,0.4)