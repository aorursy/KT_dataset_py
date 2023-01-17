# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from pathlib import Path



from sklearn.cluster import KMeans



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
KAGGLE_PATH = Path('/kaggle/input/trends-assessment-prediction')



# subject-levels

#SCN - Sub-cortical Network

#ADN - Auditory Network

#SMN - Sensorimotor Network

#VSN - Visual Network

#CON - Cognitive-control Network

#DMN - Default-mode Network

#CBN - Cerebellar Network

SL = ['SCN','ADN','SMN','VSN','CON','DMN','CBN']
sfnc = pd.read_csv(KAGGLE_PATH/'fnc.csv') #.drop('Id',axis=1)



ids = sfnc.pop('Id')



cols = sfnc.columns



sfnc.shape
group_columns={}



for c in cols:

    groupkey = c.split('(')[0] + '_' + c.split('(')[1].split('_',-1)[2]

    

    group_col_list = group_columns.get(groupkey)

    

    if group_col_list == None:

        group_col_list = [c]

    else:

        group_col_list += [c] 

    

    group_columns[groupkey] = group_col_list



# test

group_columns['SCN_SCN']
def gen_clusters(n_clusters = 3, suffix=''):

    

    sfnc_group_clusters = pd.DataFrame(ids)

    sfnc_dist_to_cluster_center = sfnc_group_clusters.copy()



    for gc in group_columns:



        X = sfnc[group_columns[gc]].values



        kmeans = KMeans(n_clusters=n_clusters, random_state=2020).fit(X)

        sfnc_group_clusters[gc] = kmeans.labels_



        #preds = kmeans.predict(sfnc[group_columns[gc]].head().values)  # ==> same as kmeans.labels

        #kmeans.cluster_centers_,



        ## euclidean distance to n cluster center 

        for cc in range(n_clusters):

            sfnc_dist_to_cluster_center[gc+'_c'+str(cc)] = (((sfnc[group_columns[gc]] - kmeans.cluster_centers_[cc])**2).sum(axis=1))**0.5



    # Test



    #sfnc_group_clusters, kmeans.cluster_centers_,

    display(sfnc_dist_to_cluster_center.head())



    sfnc_group_clusters.to_csv('sfnc_group_clusters'+suffix+'.csv',index=False)

    sfnc_dist_to_cluster_center.to_csv('sfnc_dist_to_cluster_center'+suffix+'.csv',index=False)
gen_clusters(n_clusters = 3, suffix='')
gen_clusters(n_clusters = 2, suffix='_2c')