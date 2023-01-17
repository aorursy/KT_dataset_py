# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import time 

import matplotlib.pyplot as plt

import seaborn as sns

import time
df = pd.read_csv('/kaggle/input/lish-moa/train_features.csv',index_col = 0)  

df
y = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv',index_col = 0 )

y_save = y.copy()

y
print( df['cp_type'].value_counts() )

print( df['cp_dose'].value_counts() )

print( df['cp_time'].value_counts() )
df['cp_type'] = df['cp_type'].map({'trt_cp':1.0, 'ctl_vehicle':0.0})

df['cp_dose'] = df['cp_dose'].map({'D1':0.0, 'D2':1.0})

df['cp_time'] = df['cp_time'].map({24:0.0, 48: .5 , 72:1.0})



X = df.values



df
t0 = time.time()

!pip install scikit-hubness

print(time.time()-t0,'seconds passed')
y = y_save['proteasome_inhibitor'].values

y_save['proteasome_inhibitor'].value_counts()

cutoff4sample_size = 100_000 # To quickly test idea - put small cutoff #  1_000 - 0.95 second, 5_000 - 8.79 seconds, 10_000 - 27.77,  23814 (full size) - 142.23 seconds 

t0=time.time()

from sklearn.model_selection import cross_val_score

from skhubness.neighbors import KNeighborsClassifier



# vanilla kNN

knn_standard = KNeighborsClassifier(n_neighbors=5,

                                    metric='cosine')

acc_standard = cross_val_score(knn_standard, X[:cutoff4sample_size,:], y[:cutoff4sample_size], cv=5)



# kNN with hubness reduction (mutual proximity)

knn_mp = KNeighborsClassifier(n_neighbors=5,

                              metric='cosine',

                              hubness='mutual_proximity')

acc_mp = cross_val_score(knn_mp, X[:cutoff4sample_size,:], y[:cutoff4sample_size], cv=5)



print(f'Accuracy (vanilla kNN): {acc_standard.mean():.3f}')

print(f'Accuracy (kNN with hubness reduction): {acc_mp.mean():.3f}')



print( -(t0-time.time() ) , 'seconds passed')
df_stat = pd.DataFrame()

verbose = 0 

t_previous_info_print = 0

timedelta4output_in_seconds = 3600



z = y_save.sum(axis = 0) # Calculate number if 1 in each target 

#list_selected_targets_names = list(z.sort_values(ascending = False)[:40].index) + list(z.sort_values(ascending = True)[:40].index)# Order by number of 1

list_selected_targets_names = z.sort_values(ascending = False).index



cutoff4sample_size = 100_000 # To quickly test idea - put small cutoff # for target proteasome_inhibitor :   1_000 - 0.95 second, 5_000 - 8.79 seconds, 10_000 - 27.77,  23814 (full size) - 142.23 seconds 

if cutoff4sample_size >= X.shape[0]: cutoff4sample_size = X.shape[0]

    

t00=time.time()

for i in range(len(list_selected_targets_names)):

    if 1:

        target_name = list_selected_targets_names[i]

    else:

        target_name = y_save.columns[i]

    y = y_save[target_name].values

    

    df_stat.loc[i,'Name Target'] = target_name

    

    t0=time.time()

    from sklearn.model_selection import cross_val_score

    from skhubness.neighbors import KNeighborsClassifier



    # vanilla kNN

    knn_standard = KNeighborsClassifier(n_neighbors=5,

                                        metric='cosine')

    acc_standard = cross_val_score(knn_standard, X[:cutoff4sample_size,:], y[:cutoff4sample_size], cv=5)

    df_stat.loc[i,'Score'] = acc_standard.mean()



    

    # kNN with hubness reduction (mutual proximity)

    knn_mp = KNeighborsClassifier(n_neighbors=5,

                                  metric='cosine',

                                  hubness='mutual_proximity')

    acc_mp = cross_val_score(knn_mp, X[:cutoff4sample_size,:], y[:cutoff4sample_size], cv=5)

    df_stat.loc[i,'Score Hub Reduced'] = acc_mp.mean()

    

    df_stat.loc[i,'Score Improve'] = acc_mp.mean() - acc_standard.mean()

    

    # Service things: 

    df_stat.loc[i,'#1 in target'] = y.sum()

    df_stat.loc[i,'Time (seconds)'] = np.round( -(t0-time.time() ) ,2)

    df_stat.loc[i,'Sample size'] = cutoff4sample_size

    

    if verbose > 10:

        print(f'Accuracy (vanilla kNN): {acc_standard.mean():.3f}')

        print(f'Accuracy (kNN with hubness reduction): {acc_mp.mean():.3f}')

        print( -(t0-time.time() ) , 'seconds passed')

    if (time.time() - t_previous_info_print  ) > timedelta4output_in_seconds:

        print(f'Processed {(i+1):d} targets. Passed {time.time()-t00:.3f} seconds')

        t_previous_info_print = time.time()

        

print( -(t00-time.time() ) , 'total seconds passed')    

df_stat    
df_stat['Score Improve'].describe()
plt.hist(df_stat['Score Improve'],bins = 20, label = 'Score Improve')

plt.legend()

plt.grid()

plt.show()

plt.hist(df_stat['Score Improve'].iloc[:200],bins = 20, label = 'Score Improve Top 200 targets')

plt.legend()

plt.grid()

plt.show()
df_stat.to_csv('df_stat.csv')
plt.figure(figsize = (20,6))

plt.plot(df_stat['Score Improve'], label = 'Score Improve')

plt.legend()

plt.grid()

plt.show()
df_stat.iloc[:50,:]

df_stat.iloc[50:100,:]

df_stat.iloc[100:150,:]

df_stat.iloc[150:200,:]

df_stat.iloc[200:,:]



t0=time.time()

from skhubness import Hubness

hub = Hubness(k=10, metric='cosine')

hub.fit(X[:,:])

k_skew = hub.score()

print( -(t0-time.time() ) , 'seconds passed')



print(f'Skewness = {k_skew:.3f}')

print(f'Robin hood index: {hub.robinhood_index:.3f}')

print(f'Antihub occurrence: {hub.antihub_occurrence:.3f}')

print(f'Hub occurrence: {hub.hub_occurrence:.3f}')