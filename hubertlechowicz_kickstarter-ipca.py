# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.decomposition import PCA, IncrementalPCA



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Sampling the dataset = smaller computation time.

df = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv")

df = df.sample(100000)

df = df.reset_index(drop= True)







# Lenghth of Campaign in Days

df['launched'] = pd.to_datetime(df['launched'])

df['deadline'] = pd.to_datetime(df['deadline'])

df['campaign length'] = (df.deadline-df.launched).astype('timedelta64[D]')



# Feautre Selection.

df = df.drop(['deadline','launched','ID','usd pledged'], axis=1) #Obvious drops

df = df.drop(['currency','name','pledged','goal','category','main_category'], axis=1) # Meanwhile drops, need more work to use them as features. 

df = df.drop(['country'], axis=1) # Meanwhile drops, needs cleaning.











print('State values at sample:',df.state.unique())



#State to boolean values.

for i,x in enumerate(df['state']):

     # Do wypierdolenia

    if x in ["live",'undefined']:

        df.drop([i],inplace = True)

    elif x in ["failed",'canceled',"suspended"]:

        df.at[i,'state'] = 0

    elif x == 'successful':

        df.at[i,'state'] = 1

df = df.reset_index(drop= True)



print('DataFrame shape after drops',df.shape)

print('State values as boolean:',df.state.unique())

df
X = df[['backers','usd_pledged_real','usd_goal_real','campaign length']]

y = df['state']



Y_columns = pd.get_dummies(y).columns



n_components = 2

ipca = IncrementalPCA(n_components = 2,batch_size=10)

X_ipca = ipca.fit_transform(X)



pca = PCA(n_components=2)

print(pca)

X_pca = pca.fit_transform(X)

print(X_pca)

print(X_pca.shape)

colors = ['navy', 'darkorange','green','black']



for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:

    plt.figure(figsize=(8, 8))

    for color, i, target_name in zip(colors, [0, 1, 2, 3], Y_columns):

        plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],

                    color=color, lw=2, label=target_name)



    if "Incremental" in title:

        err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()

        plt.title(title + " of kickstarter dataset\nMean absolute unsigned error "

                  "%.6f" % err)

    else:

        plt.title(title + " of kickstarter dataset")

    plt.legend(loc="best", shadow=False, scatterpoints=1)





plt.show()
