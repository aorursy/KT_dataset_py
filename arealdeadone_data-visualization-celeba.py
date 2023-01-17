# Import all the libraries
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns
import os
%matplotlib inline

!ls ../input
attr = pd.read_csv('../input/list_attr_celeba.csv')
samples = attr.sample(frac=1, random_state=42).reset_index(drop=True)
samples.head()
subset_attr = ['Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair']
attr_s = round(attr[subset_attr].describe(),2) 
attr_s
samples.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout(rect=(0, 0, 10, 10))   
fig = plt.figure(figsize = (20,5))
title = fig.suptitle("Bald Celebrities", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Bald")
ax.set_ylabel("Frequency") 
ax.text(1.2, 800, r'$\mu$='+str(round(samples['Bald'].mean(),2)), 
         fontsize=12)
freq, bins, patches = ax.hist(samples['Bald'], color='steelblue', bins=15,
                                    edgecolor='black', linewidth=1)
                                    

# Density Plot
fig = plt.figure(figsize = (20, 5))
title = fig.suptitle("Bald Celebrities", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax1 = fig.add_subplot(1,1, 1)
ax1.set_xlabel("Bald")
ax1.set_ylabel("Frequency") 
sns.kdeplot(samples['Bald'], ax=ax1, shade=True, color='steelblue')
fig = plt.figure(figsize = (20, 5))
title = fig.suptitle("Blond Hair Frequency", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Blond Hair")
ax.set_ylabel("Frequency") 
w_q = samples['Blond_Hair'].value_counts()
w_q = (list(w_q.index), list(w_q.values))
ax.tick_params(axis='both', which='major', labelsize=8.5)
bar = ax.bar(w_q[0], w_q[1], color='steelblue', 
        edgecolor='black', linewidth=1)
f, ax = plt.subplots(figsize=(30, 30))
corr = samples.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('CelebA Attributes Correlation', fontsize=14)
cols = ['Attractive', 'Bags_Under_Eyes'] 
pp = sns.pairplot(samples[cols], size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('CelebA attributes Pairwise Plots', fontsize=14)
cols = ['Blond_Hair', 'Brown_Hair']
pp = sns.pairplot(samples[cols], size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('CelebA attributes Pairwise Plots', fontsize=14)
