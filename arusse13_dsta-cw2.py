# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
UFC = pd.read_csv('../input/ufcdata/data.csv')

UFC = UFC.dropna()

UFC_1 = UFC[['Winner','B_current_win_streak','R_current_win_streak', 'B_Reach_cms', 'R_Reach_cms', 'B_age', 'R_age']]

UFC_1.head(10)
UFC_red = pd.DataFrame()

UFC_red['reach_diff'] = UFC_1['R_Reach_cms'] - UFC_1['B_Reach_cms'] #reach difference

UFC_red['age_diff'] = UFC_1['R_age'] - UFC_1['B_age'] #age difference

UFC_red['current_win_streak'] = UFC_1['R_current_win_streak'] #current win streak

UFC_red.loc[UFC_1['Winner'] == 'Red', 'Result'] = 3 #'Win'

UFC_red.loc[UFC_1['Winner'] == 'Blue', 'Result'] = 1 #'Loss'

UFC_red.loc[UFC_1['Winner'] == 'Draw', 'Result'] = 2 #'Draw'

UFC_red.head(10)

UFC_red.describe()
UFC_blue = pd.DataFrame()

UFC_blue['reach_diff'] = UFC_1['B_Reach_cms'] - UFC_1['R_Reach_cms'] #reach difference

UFC_blue['age_diff'] = UFC_1['B_age'] - UFC_1['R_age'] #age difference

UFC_blue['current_win_streak'] = UFC_1['B_current_win_streak'] #current win streak

UFC_blue.loc[UFC_1['Winner'] == 'Blue', 'Result'] = 3 #'Win'

UFC_blue.loc[UFC_1['Winner'] == 'Red', 'Result'] = 1 #'Loss'

UFC_blue.loc[UFC_1['Winner'] == 'Draw', 'Result'] = 2 #'Draw'

UFC_blue.head(10)

UFC_blue.describe()
UFC_2 = pd.concat([UFC_blue,UFC_red])

UFC_2.head(10)

UFC_2.describe()
#current_win_streak is why we need to seperate out red/bue wins/losses

#this results in inverse point generation for reach_diff and age_diff

#we will predicting win or loss
#fig_1 = plt.figure()

#ax = fig_1.add_subplot(111)

#ax.scatter(UFC_2['reach_diff'], UFC_2['age_diff'], color='darkgreen', marker='.')

#plt.show()
fig, ax = plt.subplots()

ax.scatter(UFC_2['reach_diff'], UFC_2['age_diff'], c=UFC_2['Result'], alpha=0.5)



#fig, (ax1, ax2) = plt.subplots(2)

#fig.suptitle('Vertically stacked subplots')

#ax1.scatter(UFC_2['reach_diff'], UFC_2['age_diff'], c=UFC_2['Result'])

#ax2.scatter(UFC_red['reach_diff'], UFC_red['age_diff'], c=UFC_red['Result'])
fig, ax = plt.subplots()

ax.scatter(UFC_2['reach_diff'], UFC_2['current_win_streak'], c=UFC_2['Result'], alpha=0.5)
fig, ax = plt.subplots()

ax.scatter(UFC_2['age_diff'], UFC_2['current_win_streak'], c=UFC_2['Result'], alpha=0.5)
X = UFC_2[['reach_diff','age_diff','current_win_streak']]



#plot the first three PCA dimensions

fig = plt.figure(1, figsize=(8, 6))

ax = Axes3D(fig, elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(X)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=UFC_2['Result'], cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("First three PCA directions")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])



plt.show()