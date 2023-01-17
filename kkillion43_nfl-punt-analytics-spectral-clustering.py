# Import Tools
import os
import re
import glob as glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# KungFu 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering


# Use some configs
%matplotlib inline
plt.rcParams['figure.dpi'] = 170
import os

os.chdir('../input/NFL-Punt-Analytics-Competition/')

# Read in data
data = pd.read_csv('NGS-2017-post.csv', parse_dates=True)

# Fill the Events Column Forward
data.fillna(method='ffill',inplace=True)

# Get Roles
role = pd.read_csv('play_player_role_data.csv')

# Get the Concussion data
events = pd.read_csv('video_review.csv')

# Take a summary
print(data.info(),'\n\n\n')
print(role.info(),'\n\n\n')
print(events.info(),'\n\n\n')
# First Merge the movement data with the Player Roles
posFrame = data.merge(role, how='inner', on=['Season_Year','GameKey','PlayID','GSISID'])

print(posFrame.info())
posFrame[(posFrame.PlayID.isin(events.PlayID))].sort_values('Time').head()
# Look to see what Plays Resulted in Concussions
print('Number of Plays: ', len(events.PlayID.unique()))
print(events.PlayID.unique())



# Look at all the players involved with Blocking or being Blocked
events[(events.Primary_Partner_Activity_Derived == 'Blocking') | (events.Player_Activity_Derived == 'Blocking')].head()
# The Roles and Movement during those particular plays
# Need to look at the only players involved in concussions
conFrame = posFrame[(posFrame.PlayID.isin(events.PlayID))]


print('\nList of the Plays in the Frame\n')
conFrame = conFrame[(conFrame.PlayID.isin(events.PlayID)) & (~conFrame['x'].isna())]
print(conFrame.PlayID.unique())
print('\nList of the Positions/Role in the Frame\n')
print(conFrame.Role.unique())
print('\n###############################################################\n')

print(conFrame.info())
conFrame.head()
conFrame.loc[:,['x','y']]\
.fillna(method='ffill').dropna()\
.plot(x='x', y='y',kind='scatter')

endZone1 = [0, -10]
endZone2 = [100, 110]

plt.axvspan(endZone1[0], endZone1[1], alpha=.5, lw=0, color='red')
plt.axvspan(endZone2[0], endZone2[1], alpha=.5, lw=0, color='green')
plt.xticks(np.arange(0, 101, 10))
plt.grid()
# Get the Concussion data
events = pd.read_csv('video_review.csv')

# Get Roles
role = pd.read_csv('play_player_role_data.csv')

# Run through all the data here to find all plays of interest
fileList = glob.glob(os.getcwd() + "/*.csv")

# Regex to match for NGS data
match = ['NGS']

# The Frame to build
bigFrame = pd.DataFrame()


for s in fileList:
    
    # Get all the movement NGS Data
    if (re.findall(r"(?=("+'|'.join(match)+r"))", s)):
        
        print('Processing %s...' % s.split('\\')[-1])

        df = pd.read_csv(s, 
                    parse_dates=['Time'],
                    infer_datetime_format=True,
                    dtype = {'Event' : str}) 
        
        # Carry play Event forward
        df.Event.fillna(method='ffill', inplace=True)
        
        
        # Filter for only Concussion plays
        df = df[(df.PlayID.isin(list(events.PlayID.values)))].sort_values('Time')
        
        # Now get the players involved
        df = df[df.GSISID.isin(list(events.GSISID.values))].sort_values('Time')

        
        # Fill in the Roles/Positions
        df = df.merge(role, how='inner', on=['Season_Year','GameKey','PlayID','GSISID'])
        df.reset_index(inplace=True, drop=True)
        
        # Concatenate the Data with bigFrame
        bigFrame = pd.concat([bigFrame, df], sort=False)

print('\n\nFinal DataFrame:\n\n', bigFrame.info())
bigFrame.tail()
# Sanity Check
print(sorted([int(x) for x in bigFrame.GSISID.unique()]))
print('\n\n', sorted(events.GSISID.unique()))
# Merge Events 
bigFrame1 = bigFrame.merge(events, how='inner', on=['Season_Year','GameKey','PlayID','GSISID'])
bigFrame1 = bigFrame1[bigFrame1.Primary_Partner_GSISID != 'Unclear'].dropna().apply(pd.to_numeric, errors='ignore')
bigFrame1.info()
bigFrame2 = pd.get_dummies(data=bigFrame1, columns=['Role', 
                                                    'Player_Activity_Derived',
                                                    'Primary_Impact_Type', 
                                                    'Primary_Partner_Activity_Derived'])

# Remove for clustering analytics
del bigFrame2['Event']
del bigFrame2['Turnover_Related']
del bigFrame2['Friendly_Fire']


bigFrame2.info()
# Normalize Movement data
scaler = StandardScaler()
clusterFrame = bigFrame2
clusterFrame[['dir','dis','o','x','y']] = scaler.fit_transform(bigFrame2[['dir','dis','o','x','y']])

clusterFrame.set_index('Time', inplace=True)
print(clusterFrame.info())
clusterFrame.head()
# Spectral Clustering
specCluster = SpectralClustering(n_clusters=4, eigen_solver='arpack', random_state=43, assign_labels='discretize')

clusterFrame['clusters'] = specCluster.fit_predict(clusterFrame.values)
clusterFrame.info()

clusterFrame.groupby(['clusters'])['Primary_Impact_Type_Helmet-to-helmet',
                                  'Primary_Impact_Type_Helmet-to-body']\
.agg(['sum','count','max','mean','min','first','last'])
clusterFrame.groupby(['clusters'])['Primary_Partner_Activity_Derived_Blocked',
                                  'Primary_Partner_Activity_Derived_Blocking']\
.agg(['sum','count','max','mean','min','first','last'])
clusterFrame.groupby(['clusters'])['Primary_Partner_Activity_Derived_Tackled',
                                  'Primary_Partner_Activity_Derived_Tackling']\
.agg(['sum','count','max','mean','min','first','last'])
clusterFrame.groupby(['clusters'])['Player_Activity_Derived_Tackled',
                                  'Player_Activity_Derived_Tackling']\
.agg(['sum','count','max','mean','min','first','last'])
clusterFrame.groupby(['clusters'])['Player_Activity_Derived_Blocked',
                                  'Player_Activity_Derived_Blocking']\
.agg(['sum','count','max','mean','min','first','last'])
# Look through the clusters

sumFrame = pd.DataFrame()

for cluster in [0,1,2,3]:
    interestFrame = clusterFrame[clusterFrame.clusters == cluster]

    print('\n\n')
    print('Number of Plays in Cluster %s :' % cluster, len(interestFrame['PlayID'].unique()))
    print('Plays from Cluster %s : ' % cluster, interestFrame['PlayID'].unique())
    print('Players from Cluster %s : ' % cluster, interestFrame['GSISID'].unique(), interestFrame.Primary_Partner_GSISID.unique())

    
    players = interestFrame.Primary_Partner_GSISID.unique()
    
    activityFrame = interestFrame[interestFrame.Primary_Partner_GSISID.isin([int(x) for x in players])]\
          .loc[:,['PlayID',
                  'Primary_Partner_GSISID',
                  'GSISID',
                  'Player_Activity_Derived_Blocked',
                  'Player_Activity_Derived_Blocking',             
                  'Player_Activity_Derived_Tackled',              
                  'Player_Activity_Derived_Tackling',
                  'Primary_Partner_Activity_Derived_Blocked',
                  'Primary_Partner_Activity_Derived_Blocking',
                  'Primary_Partner_Activity_Derived_Tackled',
                  'Primary_Partner_Activity_Derived_Tackling',
                  'Primary_Impact_Type_Helmet-to-helmet',
                  'Primary_Impact_Type_Helmet-to-body']]


    print('\n\n')
    results = activityFrame[activityFrame > 0].groupby(['PlayID','GSISID', 'Primary_Partner_GSISID']).last().sum()
    print(results)
    
    activityFrame.groupby(['PlayID','GSISID', 'Primary_Partner_GSISID'])['Player_Activity_Derived_Blocked',
                  'Player_Activity_Derived_Blocking',             
                  'Player_Activity_Derived_Tackled',              
                  'Player_Activity_Derived_Tackling',
                  'Primary_Partner_Activity_Derived_Blocked',
                  'Primary_Partner_Activity_Derived_Blocking',
                  'Primary_Partner_Activity_Derived_Tackled',
                  'Primary_Partner_Activity_Derived_Tackling',
                  'Primary_Impact_Type_Helmet-to-helmet',
                  'Primary_Impact_Type_Helmet-to-body']\
    .last().sum().plot(kind='bar', legend=False)
    
    plt.show()
    
    filter_cols = [col for col in clusterFrame if col.startswith('Role')]
    filter_cols.append('clusters')
    filter_cols.append('PlayID')

    interestFrame.loc[:,filter_cols].groupby(['clusters','PlayID'])\
    .last().sum().plot(kind='bar',legend=False)
    
    plt.show()
    
    sumFrame = sumFrame.append(results, ignore_index=True)
    
sumFrame.sum().plot(kind='bar')
print(sumFrame.loc[:,['Player_Activity_Derived_Tackled',              
                  'Player_Activity_Derived_Tackling',
                'Primary_Partner_Activity_Derived_Tackled',
                  'Primary_Partner_Activity_Derived_Tackling']].sum())
sumFrame.loc[:,['Player_Activity_Derived_Blocked','Player_Activity_Derived_Blocking',
                'Primary_Partner_Activity_Derived_Blocked','Primary_Partner_Activity_Derived_Blocking']].sum()
from IPython.display import HTML

HTML('<video width="560" height="315" controls> <source src="http://a.video.nfl.com//films/vodzilla/153280/Wing_37_yard_punt-cPHvctKg-20181119_165941654_5000k.mp4" type="video/mp4"></video>')