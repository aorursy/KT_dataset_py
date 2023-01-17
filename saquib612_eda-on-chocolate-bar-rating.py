#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Reading file
choco=pd.read_csv('../input/flavors_of_cacao.csv');
#getting the head of data set
choco.head()
#Information about data set
choco.info()
#Slightly Modifying the columns name
choco.columns = choco.columns.str.replace("\\n","-").str.replace(" ","-").str.strip(" ")
choco.columns
#datatypes of features
choco.dtypes
#changing cocoa-percent data
choco['Cocoa-Percent'] = choco['Cocoa-Percent'].str.replace('%','').astype(float)/100
choco.head()
choco.columns
## Look at most frequent species
choco['Specific-Bean-Origin-or-Bar-Name'].value_counts().head(10)
choco['Company-Location'].value_counts().head(10)
#Data Visualization
## Distrubution of Rating
fig, ax = plt.subplots(figsize=[16,4])
sns.distplot(choco['Rating'],ax=ax)
ax.set_title('Rating Distrubution')
## Look at distribution of Cocoa %
fig, ax = plt.subplots(figsize=[16,4])
sns.distplot(choco['Cocoa-Percent'], ax=ax)
ax.set_title('Cocoa %, Distribution')
plt.show()
choco.plot(kind='scatter', x='Rating', y='Cocoa-Percent') ;
plt.show()
choco.plot(kind='scatter',x='Rating',y='Review-Date')
## Look at boxplot over the company location,
fig, ax = plt.subplots(figsize=[6, 16])
sns.boxplot(
    data=choco,
    y='Company-Location',
    x='Rating'
)
ax.set_title('Boxplot, Rating for countries')
## Look at rating by cocao-percent
fig, ax = plt.subplots(figsize=[6, 16])
sns.boxplot(
    data=choco,
    y='Cocoa-Percent',
    x='Rating'
)
ax.set_title('Boxplot, Rating by Cocao-Percent')
## Let's define blend feature
choco['is_blend'] = np.where(
    np.logical_or(
        np.logical_or(choco['Bean-Type'].str.lower().str.contains(',|(blend)|;'),
                      choco['Company-Location'].str.len() == 1),
        choco['Company-Location'].str.lower().str.contains(',')
    )
    , 1
    , 0
)
## How many blends/pure cocoa?
choco['is_blend'].value_counts()
## What better? Pure or blend?
fig, ax = plt.subplots(figsize=[6, 6])
sns.boxplot(
    data=choco,
    x='is_blend',
    y='Rating',
)
ax.set_title('Boxplot, Rating by Blend/Pure')
choco_best_beans = choco.groupby('Broad-Bean-Origin')['Rating'] \
                        .aggregate(['mean', 'var', 'count']) \
                        .replace(np.NaN, 0) \
                        .sort_values(['mean', 'var'], ascending=[False, False])
choco_best_beans.head()
choco_best_beans = choco_best_beans.sort_values('count', ascending=False)[:20] \
                            .sort_values('mean', ascending=False)
choco_best_beans.head()
#Country producing best choclate-bar
choco_highest = choco.groupby('Company-Location')['Rating'] \
                        .aggregate(['mean', 'var', 'count']) \
                        .replace(np.NaN, 0) \
                        .sort_values(['mean', 'var'], ascending=[False, False])
choco_highest.head()
choco_highest = choco_highest.sort_values('count', ascending=False)[:20] \
            .sort_values('mean', ascending=False)
    
choco_highest.head()
