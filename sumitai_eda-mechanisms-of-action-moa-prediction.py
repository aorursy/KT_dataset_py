import plotly 

plotly.offline.init_notebook_mode (connected = True)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import seaborn as sns 

import matplotlib.pyplot as plt 

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = '../input/lish-moa'

os.listdir(path) 
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

train_features.head()
train_targets_scored.head()
train_targets_nonscored.head()
print('______________train_features________________')

train_features.info()

print('______________________________')

print('______________test_features________________')

test_features.info()

print('______________________________')

print('_____________train_targets_scored_________________')

train_targets_scored.info()

print('______________________________')

print('________________train_targets_nonscored______________')

train_targets_nonscored.info()


print("Shape of the training set: ", train_features.shape)

print('unique ids: ', len(train_features.sig_id.unique()))



print("Shape of the training set: ", test_features.shape)

print('unique ids: ', len(test_features.sig_id.unique()))
import seaborn as sns 

fig,ax= plt.subplots(1,2)

sns.countplot(train_features['cp_time'], ax = ax[0]).set_title('For Training Set')

sns.countplot(test_features['cp_time'], ax = ax[1]).set_title('For Testing Set')

plt.tight_layout()
import seaborn as sns 

fig,ax= plt.subplots(1,2)

sns.countplot(train_features['cp_type'], ax = ax[0]).set_title('For Training Set')

sns.countplot(test_features['cp_type'], ax = ax[1]).set_title('For Testing Set')

plt.tight_layout()
import seaborn as sns 

fig,ax= plt.subplots(1,2)

sns.countplot(train_features['cp_dose'], ax = ax[0]).set_title('For Training Set')

sns.countplot(test_features['cp_dose'], ax = ax[1]).set_title('For Testing Set')

plt.tight_layout()
x = train_features.drop(['sig_id'], axis=1)

corr = x.corr()

corr.style.background_gradient(cmap='coolwarm')
# drop the first column ('sig_id'), and 

df = train_targets_scored.drop(['sig_id'], axis=1).sum(axis=0).sort_values(ascending=False).reset_index()

df.columns = ['column', 'nonzero_records']

df
# plot the bar 



fig = px.bar(

    df.head(50), 

    x='nonzero_records', 

    y='column', 

    orientation='h', 

    title='Columns with the positive samples (Only top 50)', 

    height=1000, 

    width=800

)

fig.show()
# drop the first column ('sig_id') and count the 0s in 

df1 = train_targets_scored.drop(['sig_id'], axis=1).sum(axis=0).sort_values(ascending=False).reset_index()

df1.columns = ['column', '% nonzero_records']

df1['% nonzero_records'] = (df1['% nonzero_records']/len(train_targets_scored))*100

# plot the bar 



fig = px.bar(

    df1.head(50), 

    x='% nonzero_records', 

    y='column', 

    orientation='h', 

    title='Columns with the % positive samples (Only top 50) ', 

    height=1000, 

    width=800

)

fig.show()
