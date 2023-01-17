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
seed_data = pd.read_csv('/kaggle/input/seed-from-uci/Seed_Data.csv')
seed_data.shape
# this is how the dataset looks 
seed_data
feature = seed_data.columns
print(feature)
feature.drop(['target'])  
from sklearn.preprocessing import StandardScaler
x = seed_data.loc[:,feature].values
x = StandardScaler().fit_transform(x)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
seed_data_transform = pca.fit_transform(x)
principal_seed_data = pd.DataFrame(data = seed_data_transform, columns = ['principal component 1', 'principal component 2','principal component 3'])
principal_seed_data.tail()
print(pca.explained_variance_) 
print('Explained variation per principal compoents {}'.format(pca.explained_variance_ratio_))
import matplotlib.pyplot as plt
%matplotlib notebook

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of SEED DATA  Dataset",fontsize=20)
# we had numerical data thus we did not write it in quotes. ['1','2'...]
targets = [0,1,2]
colors = ['r', 'g','b']
for target, colour in zip(targets,colors):
    indicesToKeep = seed_data['target'] == target
    plt.scatter(principal_seed_data.loc[indicesToKeep, 'principal component 1']
               , principal_seed_data.loc[indicesToKeep, 'principal component 2'],
                c = colour, s = 50)

plt.legend(targets,prop={'size': 15})
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Y = ['r','g','b']

targets = [0,1,2]
colors = ['r', 'g','b']
for target, colour in zip(targets,colors):
    indicesToKeep = seed_data['target'] == target
    ax.scatter(principal_seed_data.loc[indicesToKeep, 'principal component 1']
               , principal_seed_data.loc[indicesToKeep, 'principal component 2'],
               principal_seed_data.loc[indicesToKeep, 'principal component 3'],
               c = colour, s = 50)


ax.set_xlabel('principal component 1')
ax.set_ylabel('principal component 2')
ax.set_zlabel('principal component 3')
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()

plt.legend(targets,prop={'size': 15})
