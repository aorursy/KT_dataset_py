#default kaggle code



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import spearmanr

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#read in the data as df (data frame)

df = pd.read_csv('../input/Cell-Cycle-Set.csv')



# curate to drop columns with missing values

df.dropna(inplace=True)
sns.set(color_codes=True)



fig1_1, (ax1_1, ax1_2, ax1_3) = plt.subplots(1, 3, sharey=True, figsize=(15,5))

plt.tight_layout()

plt.title('Cell cyle stages of the RNA and Protein distribution')



#mean_RNA_G1

df.mean_RNA_G1.hist(ax=ax1_1, label='RNA')

df.mean_protein_G1.hist(ax=ax1_1, label='Protein')

ax1_1.legend()

ax1_1.set_xlabel('Mean G1 Expression')

print("G1")

print("The mean of the distribution of RNA G1 is ", df.mean_RNA_G1.mean())

print("The mean of the distribution protein G1 is ", df.mean_protein_G1.mean())

print("The variance of the distribution of RNA G1 is ", df.mean_RNA_G1.var())

print("The variance of the distribution protein G1 is ", df.mean_protein_G1.var(), "\n")



#mean_RNA_S

df.mean_RNA_S.hist(ax=ax1_2, label='RNA')

df.mean_protein_S.hist(ax=ax1_2, label='Protein')

ax1_2.legend()

ax1_2.set_xlabel('Mean S Expression')

print("S")

print("The mean of the distribution of RNA S is ", df.mean_RNA_S.mean())

print("The mean of the distribution protein S is ", df.mean_protein_S.mean())

print("The variance of the distribution of RNA S is ", df.mean_RNA_S.var())

print("The variance of the distribution protein S is ", df.mean_protein_S.var(), "\n")



#mean_RNA_G2

df.mean_RNA_S.hist(ax=ax1_3, label='RNA')

df.mean_protein_G2.hist(ax=ax1_3, label='Protein')

ax1_3.legend()

ax1_3.set_xlabel('Mean G2 Expression')

print("G2")

print("The mean of the distribution of RNA G2 is ", df.mean_RNA_G2.mean())

print("The mean of the distribution protein G2 is ", df.mean_protein_G2.mean())

print("The variance of the distribution of RNA G2 is ", df.mean_RNA_G2.var())

print("The variance of the distribution protein G2 is ", df.mean_protein_G2.var(), "\n")

# Task 2



fig3,ax3 = plt.subplots()

cax = ax3.imshow(df.corr(), cmap='magma')

ax3.set_xticklabels(df.columns, rotation=90)

ax3.set_yticklabels(df.columns)

fig3.colorbar(cax)



df.corr()
#Highlighting the anomaly in the data. 

value=(df['mean_RNA_G1']<6)

df['color1']= np.where( value==True , "#f54242", "#3498db")



fig2, ax2_2 = plt.subplots(figsize=(12,10))

sns.regplot(data=df, x="mean_RNA_G1", y="mean_protein_G1", scatter_kws={'facecolors':df['color1']})



spearmanr(df.mean_RNA_G1.values, df.mean_protein_G1.values)
value=(df['mean_RNA_S']<6)

df['color2']= np.where( value==True , "#f54242", "#3498db")



fig2, ax2_3 = plt.subplots(figsize=(12,10))

sns.regplot(data=df, x="mean_RNA_S", y="mean_protein_S", scatter_kws={'facecolors':df['color2']})



spearmanr(df.mean_RNA_S.values, df.mean_protein_S.values)
value=(df['mean_RNA_G2']<6)

df['color3']= np.where( value==True , "#f54242", "#3498db")



fig2, ax2_4 = plt.subplots(figsize=(12,10))

sns.regplot(data=df, x="mean_RNA_G2", y="mean_protein_G2", scatter_kws={'facecolors':df['color3']})



spearmanr(df.mean_RNA_G2.values, df.mean_protein_G2.values)
# Week 2



# Task 1 - "cell cycle" in GOBP 



gobp = df[df.GOBP.str.contains('cell cycle')]

fig4,ax4 = plt.subplots(ncols=3, figsize=(15,5))

df.plot.scatter('mean_RNA_G1', 'mean_protein_G1', ax=ax4[0], color='k' )

ax4[0].scatter(gobp.mean_RNA_G1, gobp.mean_protein_G1, color='r', s=10., label='r={:0.2f}'.format(

            spearmanr(gobp.mean_RNA_G1.values, gobp.mean_protein_G1.values)[0]) )

df.plot.scatter('mean_RNA_S', 'mean_protein_S', ax=ax4[1], color='k')

ax4[1].scatter(gobp.mean_RNA_S, gobp.mean_protein_S, color='r', s=10.)

df.plot.scatter('mean_RNA_G2', 'mean_protein_G2', ax=ax4[2], color='k')

ax4[2].scatter(gobp.mean_RNA_G2, gobp.mean_protein_G2, color='r', s=10.)

print(len(gobp))



print(spearmanr(gobp.mean_RNA_G1.values, gobp.mean_protein_G1.values))

print(spearmanr(gobp.mean_RNA_S.values, gobp.mean_protein_S.values))

print(spearmanr(gobp.mean_RNA_G2.values, gobp.mean_protein_G2.values))
# "actin cytoskeleton organization" in GOBP

gobp2 = df[df.GOBP.str.contains('actin cytoskeleton organization')]

fig4,ax4_2 = plt.subplots(ncols=3, figsize=(15,5))

df.plot.scatter('mean_RNA_G1', 'mean_protein_G1', ax=ax4_2[0], color='k')

ax4_2[0].scatter(gobp2.mean_RNA_G1, gobp2.mean_protein_G1, color='r', s=10.)

df.plot.scatter('mean_RNA_S', 'mean_protein_S', ax=ax4_2[1], color='k')

ax4_2[1].scatter(gobp2.mean_RNA_S, gobp2.mean_protein_S, color='r', s=10.)

df.plot.scatter('mean_RNA_G2', 'mean_protein_G2', ax=ax4_2[2], color='k')

ax4_2[2].scatter(gobp2.mean_RNA_G2, gobp2.mean_protein_G2, color='r', s=10.)

print(len(gobp2))



print(spearmanr(gobp2.mean_RNA_G1.values, gobp2.mean_protein_G1.values))

print(spearmanr(gobp2.mean_RNA_S.values, gobp2.mean_protein_S.values))

print(spearmanr(gobp2.mean_RNA_G2.values, gobp2.mean_protein_G2.values))
# Task 2

gocc = df[df.GOCC.str.contains('ribosome')]

fig5,ax5 = plt.subplots(ncols=3, figsize=(15,5))

df.plot.scatter('mean_RNA_G1', 'mean_protein_G1', ax=ax5[0], color='k')

ax5[0].scatter(gocc.mean_RNA_G1, gocc.mean_protein_G1, color='r', s=10.)

df.plot.scatter('mean_RNA_S', 'mean_protein_S', ax=ax5[1], color='k')

ax5[1].scatter(gocc.mean_RNA_S, gocc.mean_protein_S, color='r', s=10.)

df.plot.scatter('mean_RNA_G2', 'mean_protein_G2', ax=ax5[2], color='k')

ax5[2].scatter(gocc.mean_RNA_G2, gocc.mean_protein_G2, color='r', s=10.)

print(len(gobp))



print(spearmanr(gocc.mean_RNA_G1.values, gocc.mean_protein_G1.values))

print(spearmanr(gocc.mean_RNA_S.values, gocc.mean_protein_S.values))

print(spearmanr(gocc.mean_RNA_G2.values, gocc.mean_protein_G2.values))
# Task 3



"""

str.split, splits all the data in the GOBP column delimited by ";" and then counts 

the occurences of each of the GOPB terms 

"""

print(df.GOBP.str.split(';',expand=True).stack().value_counts().to_string())

# Task 4

df['mean_RNA_g1s'] = (df.mean_RNA_S - df.mean_RNA_G1)

df['mean_RNA_sg2'] = (df.mean_RNA_G2 - df.mean_RNA_S)

df['mean_RNA_g2g1'] = (df.mean_RNA_G1 - df.mean_RNA_G2)

df['mean_protein_g1s'] = (df.mean_protein_S - df.mean_protein_G1)

df['mean_protein_sg2'] = (df.mean_protein_G2 - df.mean_protein_S)

df['mean_protein_g2g1'] = (df.mean_protein_G1 - df.mean_protein_G2)



# standardise

df.iloc[:,-6:] = (df.iloc[:,-6:] - df.iloc[:,-6:].mean(axis=0)) / df.iloc[:,-6:].std(axis=0)



gobp = df[df.GOBP.str.contains('cell cycle')]

gocc = df[df.GOCC.str.contains('ribosome')]

fig6,ax6 = plt.subplots(ncols=3, figsize=(15,5))

df.plot.scatter('mean_RNA_g1s', 'mean_protein_g1s', ax=ax6[0], color='k', alpha=.5)

df.plot.scatter('mean_RNA_sg2', 'mean_protein_sg2', ax=ax6[1], color='k', alpha=.5)

df.plot.scatter('mean_RNA_g2g1', 'mean_protein_g2g1', ax=ax6[2], color='k', alpha=.5)

ax6[0].scatter(gobp.mean_RNA_g1s, gobp.mean_protein_g1s, color='r', s=10., alpha=.7)

ax6[1].scatter(gobp.mean_RNA_sg2, gobp.mean_protein_sg2, color='r', s=10., alpha=.7)

ax6[2].scatter(gobp.mean_RNA_g2g1, gobp.mean_protein_g2g1, color='r', s=10., alpha=.7)

ax6[0].scatter(gocc.mean_RNA_g1s, gocc.mean_protein_g1s, color='g', s=10., alpha=.7)

ax6[1].scatter(gocc.mean_RNA_sg2, gocc.mean_protein_sg2, color='g', s=10., alpha=.7)

ax6[2].scatter(gocc.mean_RNA_g2g1, gocc.mean_protein_g2g1, color='g', s=10., alpha=.7)

fig6,ax6 = plt.subplots(ncols=3, figsize=(15,5))

df.plot.scatter('mean_RNA_g1s', 'mean_protein_g1s', ax=ax6[0], color='k', alpha=.5)

df.plot.scatter('mean_RNA_sg2', 'mean_protein_sg2', ax=ax6[1], color='k', alpha=.5)

df.plot.scatter('mean_RNA_g2g1', 'mean_protein_g2g1', ax=ax6[2], color='k', alpha=.5)

ax6[0].scatter(gobp.mean_RNA_g1s, gobp.mean_protein_g1s, color='k', s=10., alpha=.7)

ax6[1].scatter(gobp.mean_RNA_sg2, gobp.mean_protein_sg2, color='k', s=10., alpha=.7)

ax6[2].scatter(gobp.mean_RNA_g2g1, gobp.mean_protein_g2g1, color='k', s=10., alpha=.7)

ax6[0].scatter(gocc.mean_RNA_g1s, gocc.mean_protein_g1s, color='g', s=10., alpha=.7)

ax6[1].scatter(gocc.mean_RNA_sg2, gocc.mean_protein_sg2, color='g', s=10., alpha=.7)

ax6[2].scatter(gocc.mean_RNA_g2g1, gocc.mean_protein_g2g1, color='g', s=10., alpha=.7)
#find all unique GOBP terms

terms = df.GOBP.str.split(';',expand=True).stack().unique()



#new dataframe with GOBP term and its correlation

corrs = pd.DataFrame(columns = ['GOBP term', 'correlation'])





for term in terms:

    

    #create buffer to hold dataframe of the current unique term we are looking at

    buffer = df[df.GOBP.str.contains(term)]

    

    #append the new row with the GOBP term and its correlation

    corrs = corrs.append({'GOBP term':term, 'correlation':spearmanr(

        buffer.mean_RNA_G1.values, buffer.mean_protein_G1.values)[0]}, ignore_index=True)

                                 

# curate to drop columns with missing values

corrs.dropna(inplace=True)       



#show dataframe

print(corrs)



fig4_4, ax4_4 = plt.subplots(figsize=(15,15))

ax4_4.set_xlabel('Correlation')

ax4_4.set_ylabel('Cumulative values of all terms')

plt.title('Correlation for all GOBP terms')



corrs.correlation.hist(label='Correlation', bins=20)

#find all unique GOCC terms

terms = df.GOCC.str.split(';',expand=True).stack().unique()



#new dataframe with GOCC term and its correlation

corrs = pd.DataFrame(columns = ['GOCC term', 'correlation'])





for term in terms:

    

    #create buffer to hold dataframe of the current unique term we are looking at

    buffer = df[df.GOCC.str.contains(term)]

    

    #append the new row with the GOCC term and its correlation

    corrs = corrs.append({'GOCC term':term, 'correlation':spearmanr(

        buffer.mean_RNA_G1.values, buffer.mean_protein_G1.values)[0]}, ignore_index=True)

                                 

# curate to drop columns with missing values

corrs.dropna(inplace=True)       



#show dataframe

print(corrs)



fig4_4, ax4_4 = plt.subplots(figsize=(15,15))

ax4_4.set_xlabel('Correlation')

ax4_4.set_ylabel('Cumulative values of all terms')

plt.title('Correlation for all GOBP terms')



corrs.correlation.hist(label='Correlation', bins=20)
