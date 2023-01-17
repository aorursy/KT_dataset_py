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
df_train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

df_target = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

df_target_ns = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')



ys = df_target.drop('sig_id', axis = 1)

yns = df_target_ns.drop('sig_id', axis = 1)

y = pd.concat([ys,yns], axis=1)

sig_ids = df_target['sig_id']

print(y.shape)
print(y.shape)

temp = y.sum()

tokeep = temp[temp != 0].index



ny = y[tokeep]

print(ny.shape)
def convert_binary_string(x,cols):

    

    """Convert list of targets by actual target"""

    sub = [cols[k] for k in np.where(x.values == 1)]

    if not len(sub[0]):

        sub = [['no_target']]

        

    return '--'.join(sub[0])



cols = ny.columns

targets = ny.apply(lambda x: convert_binary_string(x,cols), axis=1)



vcount = targets.value_counts()



# Remove the samples without target

vcount = vcount.iloc[1:]



for i in range(len(vcount)):

    print(f'{vcount.index[i]} -> {vcount.values[i]} occurences')

import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(

    go.Bar(

        x = vcount.index,

        y = vcount.values

    )

)

fig.show()
validated_unique_samples = vcount.groupby(vcount<=7).sum()/vcount.sum()

fig = go.Figure()

fig.add_trace(

    go.Bar(

        x = validated_unique_samples.index,

        y = validated_unique_samples.values

    )

)

fig.show()
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt



plt.figure(figsize = (25,15))

for i in range(25):

    cat = vcount[(vcount > 10) & (vcount < 20)].index[i]

    X = df_train.loc[np.where(targets == cat)].drop(['sig_id','cp_type','cp_time','cp_dose'],axis=1)

    pca = PCA()

    Xpca = pca.fit_transform(X)

    plt.subplot(5,5,i+1)

    plt.title(cat)

    plt.scatter(Xpca[:,0],Xpca[:,1])

plt.show()