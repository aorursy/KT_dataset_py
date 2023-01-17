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



df = pd.read_csv('/kaggle/input/lish-moa/train_features.csv',index_col = 0)  

df
df_test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv',index_col = 0)  

df_test
y = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv',index_col = 0 )

y
y_additional = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv',index_col = 0 )

y_additional
z = y.sum(axis = 0)

print( (z==6).sum() , (z==7).sum(),   (z==12).sum(),  (z==13).sum(), (z==17).sum(),  (z==18).sum(),  )

z.sort_values()[:40] #.value_counts()
z.sort_values()[-40:] #.value_counts()
from sklearn.metrics import roc_auc_score

df_features_stat = pd.DataFrame()



df_stat = pd.DataFrame()

i=0

t0 = time.time()

set_best_predictor = set()

for t in y.columns:

    i+=1

    print(i, t, time.time() -t0,'secs passed')

    for f in df.columns: #['g-100']:

        if 'g-' not in f:

            continue

        try:

            r = roc_auc_score(y[t], df[f] )

        except:

            continue

        #print(r, )

        df_stat.loc[t , f ] =  np.abs(r-0.5)+0.5 

        

    set_best_predictor.add(df_stat.columns[df_stat.loc[t,:].argmax()])

    print(i, 'Best Auc:',df_stat.loc[t,:].max(), 'For Gen:',  df_stat.columns[df_stat.loc[t,:].argmax()], 'LenSetBestGenes:', len( set_best_predictor ) )

    

    #if i > 2: break   

        

v = df_stat.max(axis = 1)

v.sort_values(inplace = True, ascending = False) # ('rocauc Abs', inplace = True, ascending = False)



plt.figure(figsize = (15,6))

plt.plot(v.values,'*')        

plt.show()



v
df_association = pd.DataFrame()



for t in df_stat.index:

    df_association.loc[t,'#1'] = y[t].sum()

    v = df_stat.loc[t,:]

    v.sort_values(inplace = True, ascending = False)

    for i in range(8):

        df_association.loc[t,'Top'+str(i+1)+'Gen'] = v.index[i]

    #for i in range(1):

    df_association.loc[t,'WorstGen'] = v.index[-1]

    for i in range(8):

        df_association.loc[t,'Top'+str(i+1)+'Auc'] = v.iloc[i]

    #for i in range(1):

    df_association.loc[t,'WorstAuc'] = v.iloc[-1]



df_association.sort_values('Top1Auc',inplace = True, ascending = False )#]



df_association    
df_association.head(30)
df_association[ ['#1', 'Top1Gen','Top1Auc', 'Top2Gen','Top2Auc',]].head(30) # ,inplace = True, ascending = False )#]

df_association.tail(30)
#  Genes which best predict several targets 

df_association[ 'Top1Gen'].value_counts().head(20)
# What targets are predicted by g-392 which is best predictor for 8 targets ( that gene is top is that respect)  

m = df_association[  'Top1Gen' ] == 'g-392'

df_association[m]
# What targets are predicted by g-178 which is best predictor for 5  targets ( that gene is subtop is that respect) 

m = df_association[  'Top1Gen' ] == 'g-178'

df_association[m]
# What targets are predicted by g-75 which is best predictor for 5  targets ( that gene is subtop is that respect) 

m = df_association[  'Top1Gen' ] == 'g-75'

df_association[m]


plt.figure(figsize = (20,6))

for f in ['Top1Auc','Top2Auc','Top5Auc']:

    plt.plot( df_association[f].values, label = f )

plt.legend()

plt.grid()

plt.show()
