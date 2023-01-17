# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.set_option('display.max_colwidth', -1)
path = '/kaggle/input/data-science-bowl-2019/'
test = pd.read_csv(path+'test.csv')

train = pd.read_csv(path+'train.csv')

train_labels = pd.read_csv(path+'train_labels.csv')

spcs = pd.read_csv(path+'specs.csv')

sub = pd.read_csv(path+'sample_submission.csv')
train.head(1)
print('# unique installation_id:', len(train.installation_id.unique()))

print('# unique game_session:', len(train.game_session.unique()))

print('# unique event_id:', len(train.event_id.unique()))

print('\n')

print('# unique event_code:', len(train.event_code.unique()))

print('# unique title:', len(train.title.unique()))
train.title.unique()
plt.figure(figsize=[5,4])

plt.title('TYPE')

sns.countplot(train.type)

_=plt.xticks(rotation=90)
plt.figure(figsize=[5*4,6])

for i,event_type in enumerate(train.type.unique(),1):

    plt.subplot(1,4,i)

    plt.title(event_type)

    sns.countplot(train[train.type==event_type].title)

    _=plt.xticks(rotation=90)

plt.tight_layout()
plt.figure(figsize=[5*5,5])

for i,event_type in enumerate(train.type.unique(),1):

    plt.subplot(1,4,i)

    plt.title(event_type)

    df = train[train.type==event_type]

    sns.countplot(df.event_code,order=sorted(df.event_code.unique()))

    _=plt.xticks(rotation=90)

plt.tight_layout()

del df
for event_type in train.type.unique():

    if event_type=='Clip':

        continue

    df = train[train.type==event_type]

    N = len(df.title.unique())

    W = min(5,N)

    H = N//5+1;

    plt.figure(figsize=[5*W,4*H])

    for i,title in enumerate(df.title.unique(),1):

        plt.subplot(H,W,i)

        plt.title(title)

        sns.countplot(df[df.title==title].event_code,order=sorted(df[df.title==title].event_code.unique()))

        _=plt.xticks(rotation=90)

    plt.tight_layout()

del df
for world in train.world.unique():

    print('WORLD:',world)

    print(train[train.world==world].title.unique())
sns.distplot(train.loc[:,['installation_id','game_session']].groupby('installation_id').nunique().game_session,bins=np.linspace(0,1300,21),kde=False,norm_hist=False,hist_kws={'rwidth':0.9})

plt.yscale('log')
sns.distplot(train.loc[:,['installation_id','game_session']].groupby('installation_id').nunique().game_session,bins=np.linspace(0,20,21),kde=False,norm_hist=False,hist_kws={'rwidth':0.9})

#plt.yscale('log')
sns.distplot(train.loc[:,['installation_id','game_session','event_id']].groupby(['installation_id','game_session']).count().event_id,bins=np.linspace(2,3502,36),kde=False,norm_hist=False,hist_kws={'rwidth':0.9})

plt.yscale('log')
sns.distplot(train.loc[:,['installation_id','game_session','event_id']].groupby(['installation_id','game_session']).count().event_id,bins=np.linspace(2,102,101),kde=False,norm_hist=False,hist_kws={'rwidth':0.9})

#plt.yscale('log')
plt.figure(figsize=[12,6])

plt.title('NUMBER INCORRECT')

sns.countplot(train_labels.num_incorrect,order=range(90))

plt.yscale('log')

_,_=plt.xticks(ticks = range(0,90,5), labels = range(0,90,5))
plt.figure(figsize=[5*2,4])

plt.subplot(1,2,1)

plt.title('ACCURACY GROUP')

_=sns.countplot(train_labels.accuracy_group,order=range(4))

plt.subplot(1,2,2)

plt.title('ACCURACY')

_=sns.distplot(train_labels.accuracy,kde=False, norm_hist=False, bins=np.linspace(0,1,21), hist_kws={'rwidth' : 0.8})