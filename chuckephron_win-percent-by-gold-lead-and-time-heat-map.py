import numpy as np

import pandas as pd

import time, sys, os, re, ast, itertools

import matplotlib.pyplot as plt

from collections import Counter

import seaborn as sns
def genHeatGraph(df, plots, dx, dy):

    df2= []

    for index, row in df.iterrows():

        gvals= [10000 if a > 10000 else -10000 if a < -10000 else int(int(a/dy)*dy) for a in ast.literal_eval(row.golddiff)]

        lengval= min(60+1,len(gvals))

        

        if row.bResult==1: # READ WINNER and LOSER GOLD DIFFERENCE VALUES

            for g in range(1, lengval): df2.append([int(int((g-1)/dx)*dx+dx), gvals[g], 1])

            for g in range(1, lengval): df2.append([int(int((g-1)/dx)*dx+dx), -gvals[g], 0])      

        elif row.rResult==1: 

            for g in range(1, lengval): df2.append([int(int((g-1)/dx)*dx+dx), -gvals[g], 1])

            for g in range(1, lengval): df2.append([int(int((g-1)/dx)*dx+dx), gvals[g], 0])

        else: print('Error. Bad Result'); continue      



    df2= pd.DataFrame(df2); df2.columns= ['Time (min)', 'Gold Lead', 'Win Percent']

    numgames= [int(x/10) for x in df2.groupby(['Time (min)']).size().reset_index(name='counts')['counts']] # Numer of Games for second x-Axis

    df2= df2.groupby(['Time (min)', 'Gold Lead']).apply(lambda x: x['Win Percent'].sum()/len(x)).reset_index(name='Win Percent') # Derive Win Percents

    df2['Win Percent']= df2['Win Percent'].map(lambda x: int(x*100))

    df2= df2[df2['Gold Lead']>=0] # Remove negative values (mirrors positive values)



    df2= df2.pivot('Gold Lead', 'Time (min)', 'Win Percent') # GRAPH HEAT MAP

    yticknames=['' if (k/500)%2==1 else str(int(k/1000.0))+'k' for k in range(0,10000+1,int(dy))]

    sns.heatmap(df2, vmin=0, vmax=100, yticklabels=yticknames, ax=plots, annot=True, cbar=False, fmt='g', cmap='viridis').invert_yaxis()

        

    ax2= plots.twiny() # Number of Games to second x-Axis

    ax2.set_xlim(plots.get_xlim())

    ax2.set_xticks([x+0.5 for x in range(int(dx/2),60,int(dx))]+[60])

    ax2.set_xticklabels([int(x) for x in numgames])

    ax2.set_xlabel(r'Number of Games')

    ax2.grid(False)

    

    plots.set_title('Win Percent by Time and Gold Lead ('+str(df.shape[0])+' Total Games)', y=1.06) 

    return plots
df= pd.read_csv('../input/LeagueofLegends.csv')

region= df.League.unique()

year= df.Year.unique()

dx, dy= 5.0, 500.0 # Time and Gold Buckets

df= df.loc[(df.League.isin(region) & df.Year.isin(year)),:].drop_duplicates()    

p, plots= plt.subplots(nrows=1, ncols=1, figsize=(20,10))

plots= genHeatGraph(df, plots, dx, dy)