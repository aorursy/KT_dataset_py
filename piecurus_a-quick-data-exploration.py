# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/pml-training.csv',index_col=[4])

# removes the first columns: seems to be the index

train.drop(train.columns[0],axis=1,inplace=True)

# removes columns which 51% of the data is NaN

train.dropna(axis=1,thresh=int(0.51*train.shape[0]),inplace=True)

train.head(3)
sns.factorplot('new_window',data = train,kind='count')
sns.factorplot('user_name',data = train,kind='count')
plt.plot(train['num_window'].values)

plt.ylabel('Num Window')
sns.factorplot('classe',data = train,kind='count')
def get_data_from(data,user_name):

    this_user_data = data[data['user_name']==user_name]

    unique_classes = np.unique(this_user_data['classe'])

    this_user_labels = this_user_data['classe'].map(lambda l: np.where(l==unique_classes)[0][0]).values

    this_user_data.drop('classe',axis=1,inplace=True)

    return (this_user_data,this_user_labels)



def get_data_from_position(data,position):

    # position can be: belt, arm, dumbbell, forearm

    return data.ix[:,[c.split('_')[1]==position or c.endswith(position) 

                                    for c in list(data.columns)]]
(adelmo_train,adelmo_labels) = get_data_from(train,'adelmo')

adelmo_belt = get_data_from_position(adelmo_train,'belt')

adelmo_arm = get_data_from_position(adelmo_train,'arm')

adelmo_dumbbell = get_data_from_position(adelmo_train,'dumbbell')

adelmo_forearm = get_data_from_position(adelmo_train,'forearm')
from sklearn.manifold import TSNE



tsne_belt = TSNE(2).fit_transform(adelmo_belt)

tsne_arm = TSNE(2).fit_transform(adelmo_arm)

tsne_dumbbell = TSNE(2).fit_transform(adelmo_dumbbell)

tsne_forearm = TSNE(2).fit_transform(adelmo_forearm)
plt.figure(figsize=(14,40))

#Belt

plt.subplot(411)

plt.scatter(tsne_belt[:,0],tsne_belt[:,1],c=adelmo_labels,cmap='plasma')

cbar = plt.colorbar()

cbar.set_ticks(range(5))

cbar.set_ticklabels(['A','B','C','D','E'])

plt.title('Adelmo Belt')

#Arm

plt.subplot(412)

plt.scatter(tsne_arm[:,0],tsne_arm[:,1],c=adelmo_labels,cmap='plasma')

cbar = plt.colorbar()

cbar.set_ticks(range(5))

cbar.set_ticklabels(['A','B','C','D','E'])

plt.title('Adelmo Arm')

#Dumbbell

plt.subplot(413)

plt.scatter(tsne_dumbbell[:,0],tsne_dumbbell[:,1],c=adelmo_labels,cmap='plasma')

cbar = plt.colorbar()

cbar.set_ticks(range(5))

cbar.set_ticklabels(['A','B','C','D','E'])

plt.title('Adelmo Dumbbell')

#Forearm

plt.subplot(414)

plt.scatter(tsne_forearm[:,0],tsne_forearm[:,1],c=adelmo_labels,cmap='plasma')

cbar = plt.colorbar()

cbar.set_ticks(range(5))

cbar.set_ticklabels(['A','B','C','D','E'])

plt.title('Adelmo Forearm')
(charles_train,charles_labels) = get_data_from(train,'charles')

charles_belt = get_data_from_position(charles_train,'belt')

charles_arm = get_data_from_position(charles_train,'arm')

charles_dumbbell = get_data_from_position(charles_train,'dumbbell')

charles_forearm = get_data_from_position(charles_train,'forearm')
tsne_belt = TSNE(2).fit_transform(charles_belt)

tsne_arm = TSNE(2).fit_transform(charles_arm)

tsne_dumbbell = TSNE(2).fit_transform(charles_dumbbell)

tsne_forearm = TSNE(2).fit_transform(charles_forearm)
plt.figure(figsize=(14,40))

#Belt

plt.subplot(411)

plt.scatter(tsne_belt[:,0],tsne_belt[:,1],c=charles_labels,cmap='plasma')

cbar = plt.colorbar()

cbar.set_ticks(range(5))

cbar.set_ticklabels(['A','B','C','D','E'])

plt.title('Charles Belt')

#Arm

plt.subplot(412)

plt.scatter(tsne_arm[:,0],tsne_arm[:,1],c=charles_labels,cmap='plasma')

cbar = plt.colorbar()

cbar.set_ticks(range(5))

cbar.set_ticklabels(['A','B','C','D','E'])

plt.title('Charles Arm')

#Dumbbell

plt.subplot(413)

plt.scatter(tsne_dumbbell[:,0],tsne_dumbbell[:,1],c=charles_labels,cmap='plasma')

cbar = plt.colorbar()

cbar.set_ticks(range(5))

cbar.set_ticklabels(['A','B','C','D','E'])

plt.title('Charles Dumbbell')

#Forearm

plt.subplot(414)

plt.scatter(tsne_forearm[:,0],tsne_forearm[:,1],c=charles_labels,cmap='plasma')

cbar = plt.colorbar()

cbar.set_ticks(range(5))

cbar.set_ticklabels(['A','B','C','D','E'])

plt.title('Charles Forearm')