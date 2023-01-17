import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
% matplotlib inline
import seaborn as sns

from IPython.display import display
def lahman_year_team(inputdf,yr='2012',droplist=['yearID','G','Ghome','W','L','Rank']):
    '''
    function which obtains numeric-only data from Lahman Teams data set and selects year
    inputdf    : Lahman Teams.csv, DataFrame
    yr         : year to select, int
    droplist   : static features to omit from returned data, list of str
    returns    : numeric Teams features with team_id as index, DataFrame
    '''
    df = inputdf.copy()
    # select numeric data only for PCA
    numericdf = df.select_dtypes(exclude=['object'])
    # assign team_ID as index
    numericdf.set_index(df['teamID'].values,inplace=True)
    # filter by year
    numericdf = numericdf[numericdf.yearID==yr]
    # drop constant features, where value is dependent or does not vary by player/team performance
    numericdf.drop(droplist,axis=1,inplace=True)
    print('Lahman numeric feature team results {}:'.format(yr))
    return numericdf 

teamsdf = pd.read_csv('../input/Teams.csv')
droplist=['yearID','G','Ghome','W','L','Rank']
teams12 = lahman_year_team(teamsdf,yr=2012,droplist=droplist)    
# teams12.columns
teams12.head()

def center_scale(X):
    '''
    returns : X features centered by column mean and scaled by column std, df
    '''
    return (X-np.mean(X))/np.std(X)

def pca(inputdf):
    '''
    function which computes largest variance directions (loading vectors) and principal components (score vectors)
    inputdf    : features to compute variance explained
    returns    : loading vectors, score vectors as PCs, variance explained as eigenvals    
    '''
    df = inputdf.copy()
    # step 1: center/scale the features
    C = center_scale(df)
    print('Shape of centered features matrix = {}'.format(C.shape))
    # step 2: compute cov of tranpsose of centered features
    cov = np.cov(C.T)
    print('shape of covariance matrix = {}'.format(cov.shape))
    # step 3: compute the PC loading vectors (direction of largest variacne in features space)
    eigvals,eigvecs = np.linalg.eig(cov)
    print('shape of eigenvalues, eigenvectors = {}, {}'.format(eigvals.shape,eigvecs.shape))
    loadingheaders = ['L'+str(i) for i in range(1,len(df.columns)+1)]
    # eigvecs are loadings 
    loadingdf = pd.DataFrame(eigvecs,columns=loadingheaders,index=df.columns).astype(float)
    print('shape of loadings df = {}'.format(loadingdf.shape))
    print('Top 5 PC loading vectors (direction of largest variation in feature-space):')
    display(loadingdf.loc[:,:'L5'])
    # step 4: compute score vectors as Principal Components (where scores are features C projected onto loading vectors)
    scorematrix = loadingdf.values.T.dot(C.T)
    scoreheaders = ['PC'+str(i) for i in range(1,len(C.columns)+1)]
    scoredf = pd.DataFrame(scorematrix.T,index=C.index,columns=scoreheaders)
    display(scoredf.head())
    return loadingdf,scoredf,eigvals


loadingdf,scoredf,eigvals = pca(teams12)

def pve(eigvals):
    '''
    function which computes percent variance explained (PVE), cumulative PVE of all PCs
    inputdf     : numeric features X with named indices, DataFrame
    eigvals     : eigenvalues resulting from principal components analyis, are the corresponding variance explained of ea. PC
    '''
    with plt.style.context('seaborn-white'):
        fig,ax = plt.subplots(figsize=(14,8))
        var_total = eigvals.sum()
        # compute proportional variance explained per PC
        pve = eigvals/var_total
        # compute cum. variance explained per PC
        cumpve = np.cumsum(pve)
        x = [i for i in range(1,len(eigvals)+1)]
        ax.set_xticks(x)
        ax.plot(x,pve,label='PVE')
        ax.plot(x,cumpve,label='PVE_cumulative')
        ax.set(title='Percent Variance Explained by Principal Components',
              xlabel='PC',ylabel='Variance Explained')
        # ref lines
        hlinecolor='0.74'
        ax.axhline(y=eigvals[0]/eigvals.sum(),linestyle='dotted',color=hlinecolor)
        ax.axhline(y=0,linestyle='dotted',color=hlinecolor)
        ax.axhline(y=1,linestyle='dotted',color=hlinecolor)
        ax.legend(loc='best')
pve(eigvals)

np.cumsum(eigvals/eigvals.sum())
(eigvals/eigvals.sum())
def lg_ranks(inputdf,year):
    '''
    function which displays team end of season results
    inputdf    : Lahman database Teams.csv, DataFrame
    year       : year to filter, int
    '''
    df = inputdf.copy()
    algrp = df[(df.yearID==year)&(df.lgID=='AL')].groupby(['teamID','lgID','divID','W','L'],as_index=False).agg({'Rank':'last'}).sort_values(['Rank','lgID','divID'])
    nlgrp = df[(df.yearID==year)&(df.lgID=='NL')].groupby(['teamID','lgID','divID','W','L'],as_index=False).agg({'Rank':'last'}).sort_values(['Rank','lgID','divID'])
    print('{} Final MLB Team Standings:'.format(year))
    return algrp,nlgrp

def biplot(loadingdf,scoredf,loading_color,score_color,score_axlim=7.5,load_axlim=7.5,load_arrows=4):
    '''
    function which computes biplot of PC scores, loadings
    scoredf    : matrix of PC score vectors, used tp display how indices are projected onto PC loading vectors, DataFrame
    loadingdf  : matrix of PC loading vectors from centered, std'd features, used to show actual direction of PC1 and PC2 2D vectors, DataFrame
    _color     : matplotlib line colors for corresponding loading vectors, score projection points, str
    '''
    with plt.style.context('seaborn-white'):
        f = plt.figure(figsize=(14,14))
        ax0 = plt.subplot(111)
        # plot the first two score vectors, as annotations, of teamID indices (PC1,PC2 are orhogonal to ea. other)
        for teamid in scoredf.index:  
            ax0.annotate(teamid,(scoredf['PC1'][teamid],-scoredf['PC2'][teamid]),ha='center',color=score_color)
        score_axlim = score_axlim
        ax0.set(xlim=(-score_axlim,score_axlim),ylim=(-score_axlim,score_axlim),
               )
        ax0.set_xlabel('Principal Component 1',color=score_color)
        ax0.set_ylabel('Principal Component 2',color=score_color)
        # add reference lines through origin
        ax0.hlines(y=0,xmin=-score_axlim,xmax=score_axlim,linestyle='dotted',color='grey')
        ax0.vlines(x=0,ymin=-score_axlim,ymax=score_axlim,linestyle='dotted',color='grey')
        # plot PC1 and PC2 loadings (two directions in features space with largest variation) as reference vectors
        ax1 = ax0.twinx().twiny()
        ax1.set(xlim=(-load_axlim,load_axlim), ylim=(-load_axlim,load_axlim),
               )
        ax1.tick_params(axis='y',color='red')
        ax1.set_xlabel('Principal Component Loading Weights',color=loading_color)
        # plot first two PC loading vectors (as loadingdf.index annotations)
        offset_scalar=1.175
        for feature in loadingdf.index: 
            ax1.annotate(feature,(loadingdf['L1'].loc[feature]*offset_scalar,-loadingdf['L2'].loc[feature]*offset_scalar),color=loading_color)
        # display first fourPCs as arrows
        for i in range(0,load_arrows):
            ax1.arrow(x=0,y=0,dx=loadingdf['L1'][i],dy=-loadingdf['L2'][i],head_width=0.0075,shape='full')
biplot(loadingdf,scoredf,loading_color='red',score_color='blue',score_axlim=8.5,load_axlim=.6,load_arrows=len(loadingdf.columns))        
ALrankdf,NLrankdf = lg_ranks(teamsdf,2012)
display(ALrankdf)
display(NLrankdf)
    