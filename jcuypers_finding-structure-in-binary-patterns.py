# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load some libraries for added fun

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import mixture 
# Let's set a seed, so results become reproducable 

seed = 41;

np.random.seed(seed)
# Let's load data and check if we got we we needed

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



print (df_train.shape)

print (df_test.shape)
# the numbers of components 

n_comp = 128

n_iter = 100



# Calculate the GaussianMixtures (or you could alternatively use BayesianGaussianMixture)

# In this example we split up the bit patterns in half as a comparison

# BE AWARE: this my take a while in a kaggle kernel.... (reduce groups and or range of bits to just test the basics)

X = df_train.iloc[:,10:178].values

Y = df_train.y.values

gmm = mixture.GaussianMixture(n_components=n_comp, covariance_type='full',random_state=seed,max_iter=n_iter).fit(X,Y)



X2 = df_train.iloc[:,178:366].values

Y = df_train.y.values

gmm2 = mixture.GaussianMixture(n_components=n_comp, covariance_type='full',random_state=seed,max_iter=n_iter).fit(X2,Y)
# Predict the group to which a certain sample belongs

grp_pred = gmm.predict(X)

grp_pred2 = gmm2.predict(X2)



# Put the results in dataframes for easier manipulation

grp1 = pd.DataFrame({'grp_pred': grp_pred})

grp2 = pd.DataFrame({'grp_pred': grp_pred2})
# Helper function for average and median per group

def generate_avg_and_median(grp):

    avg_grp = {}

    median_grp = {}

    

    for i in range(1,n_comp+1):

        indices = grp[grp['grp_pred']==i].index

        avg_grp[i] = df_train.iloc[indices].y.mean()

        median_grp[i] = df_train.iloc[indices].y.median()

    

    return avg_grp,median_grp
# Generate some stats per group

avgs_grp1, median_grp1 = generate_avg_and_median(grp1)

avgs_grp2, median_grp2 = generate_avg_and_median(grp2)
# Plot a pretty picture and save it

# You can modify it to overlay whatever stats you want.



plt.figure(figsize=(25,20))



plt.subplot(121)

plt.ylim(-1, n_comp+1)

plt.scatter(df_train.y,grp_pred,s=5)

for i in range(1,n_comp):

    #plt.axvline(avgs_grp1[i],c='r',alpha=0.1)

    plt.axvline(median_grp1[i],c='b',alpha=0.1)

  

plt.subplot(122)

plt.ylim(-1, n_comp+1)

plt.scatter(df_train.y,grp_pred2,s=5)

for i in range(1,n_comp+1):

   #plt.axvline(avgs_grp2[i],c='r',alpha=0.1)

   plt.axvline(median_grp2[i],c='b',alpha=0.1)



plt.show()



plt.savefig('binary_groups.png')
