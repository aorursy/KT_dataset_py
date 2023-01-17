# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
path = '/kaggle/input/learn-together/'

train_df = pd.read_csv(path+'train.csv',index_col='Id')
train_df.describe().T
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score
X_train,X_val, y_train,y_val = train_test_split(train_df.drop('Cover_Type',axis=1),train_df['Cover_Type'],random_state=10,test_size=0.3)
def print_score(m):

    res= [accuracy_score(y_train,m.predict(X_train),),accuracy_score(y_val,m.predict(X_val))]

    print(res)
rf = RandomForestClassifier(n_jobs=-1,random_state=10)

rf.fit(X_train,y_train)

print_score(rf)
def rf_feat_importances(m,df):

    return pd.DataFrame({'cols':df.columns.tolist(),'importances':m.feature_importances_}).sort_values('importances',ascending=False)
fi = rf_feat_importances(rf,X_train)

fi[:10]
def plot_fi(fi): 

    fi.plot('cols','importances','barh',figsize=(12,7),legend=False)
plot_fi(fi[:30])
to_keep = fi[fi['importances'] > 0.005]['cols']

df_keep = train_df[to_keep]
X_train,X_val,y_train,y_val = train_test_split(df_keep,train_df['Cover_Type'],test_size=0.3,random_state=10)
rf_2 = RandomForestClassifier(n_jobs=-1,random_state=10)

rf_2.fit(X_train,y_train)

print_score(rf_2)
fi_2 = rf_feat_importances(rf_2,X_train)
plot_fi(fi_2)
import scipy
from scipy.cluster import hierarchy as hc

corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(16,12))

dendrogram = hc.dendrogram(z,labels=df_keep.columns,orientation='left',leaf_font_size=16)

plt.show()
from pdpbox import pdp

from plotnine import *
X_train,X_valid,y_train,y_valid = train_test_split(train_df.drop('Cover_Type',axis=1),train_df['Cover_Type'],test_size=0.3,random_state=10)

m = RandomForestClassifier(n_estimators=40, n_jobs=-1,random_state=10)

m.fit(X_train,y_train)
def get_sample(df,n):

    idxs = sorted(np.random.permutation(len(df))[:n])

    return df.iloc[idxs].copy()
x = get_sample(X_train,500)
p = pdp.pdp_isolate(m, x, x.columns, 'Elevation')

pdp.pdp_plot(p,'Elevation',plot_lines=True,)

plt.show()
p = pdp.pdp_isolate(m, x, x.columns, 'Horizontal_Distance_To_Roadways')

pdp.pdp_plot(p,'Horizontal_Distance_To_Roadways',plot_lines=True,)

plt.show()
pi = pdp.pdp_interact(m,x,x.columns,['Elevation','Aspect'])

pdp.pdp_interact_plot(pi,['Elevation','Aspect'])

plt.show()
pi = pdp.pdp_interact(m,x,x.columns,['Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points'],n_jobs=-1)

pdp.pdp_interact_plot(pi,['Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points'])

plt.show()