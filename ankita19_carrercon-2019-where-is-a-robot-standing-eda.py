# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/X_test.csv')

x_train = pd.read_csv('../input/X_train.csv')

y_train = pd.read_csv('../input/y_train.csv')
x_train.head(5)
x_train.info()
y_train.info()
y_train.head(5)
x_train.row_id.nunique()
x_train.series_id.nunique()
x_train.measurement_number.nunique()
y_train.head(5)
y_train.groupby('series_id').first()['surface'].value_counts()
import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



plt.figure(figsize=(15,8))

sns.countplot( x = 'surface', data = y_train)
plt.figure(figsize=(15,8))

sns.boxplot( x ='surface',y = 'group_id', data = y_train)
plt.figure(figsize=(15,8))

sns.boxplot( x ='surface',y = 'series_id', data = y_train, palette = 'rainbow')
fig, ax = plt.subplots(3, 3, figsize = (16, 12))

sns.distplot(x_train['angular_velocity_X'], kde= False, bins=30, ax=ax[0][0])

sns.distplot(x_train['angular_velocity_Y'], kde= False, bins=30, ax=ax[0][1])

sns.distplot(x_train['angular_velocity_Z'], kde= False, bins=30, ax=ax[0][2])

sns.distplot(x_train['linear_acceleration_X'], kde= False, bins=30, ax=ax[1][0])

sns.distplot(x_train['linear_acceleration_Y'], kde= False, bins=30, ax=ax[1][1])

sns.distplot(x_train['linear_acceleration_Z'], kde= False, bins=30, ax=ax[1][2])

sns.distplot(x_train['orientation_X'], kde= False, bins=30, ax=ax[2][0])

sns.distplot(x_train['orientation_Y'], kde= False, bins=30, ax=ax[2][1])

sns.distplot(x_train['orientation_Z'], kde= False, bins=30, ax=ax[2][2])
x_train_2 = x_train.drop(['row_id','series_id', 'measurement_number'], axis=1)
x_train_2 .head(5)
sns.pairplot(x_train_2)
#Function to calculate the norm of a three element vector so as to make a velocity column 

def vector_norm(x,y,z,df):

    return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)



#now combine all the x, y and z dimensions of velocity and acceleration to make 1 column 

x_train['angular_velocity_norm'] =vector_norm('angular_velocity_X',

                                                'angular_velocity_Y',

                                                'angular_velocity_Z',x_train)



x_train['linear_acceleration_norm'] =vector_norm('linear_acceleration_X',

                                                'linear_acceleration_Y',

                                                'linear_acceleration_Z',x_train)



#group by series id to get an avergae value of for every series id

new_x_train = x_train.groupby('series_id')['angular_velocity_norm','linear_acceleration_norm'].mean()

new_x_train.head(5)
new_x_train  = pd.DataFrame(new_x_train).reset_index()

new_x_train.head(5)

new_x_train.columns = ['serie_id','avg_velocity','avg_acceleration']

new_x_train['surface'] = y_train.surface

new_x_train['group_id'] = y_train.group_id



new_x_train.head(8)
fig, ax = plt.subplots(1, 2, figsize = (16, 12))

plt.figure(figsize=(15,8))

sns.boxplot( x ='surface',y = 'avg_velocity', data = new_x_train, palette = 'rainbow', ax=ax[0])

sns.boxplot( x ='surface',y = 'avg_acceleration', data = new_x_train, palette = 'rainbow', ax=ax[1])
fig, ax = plt.subplots(1, 2, figsize = (16, 12))

plt.figure(figsize=(15,8))

sns.barplot( x ='surface',y = 'avg_velocity', data = new_x_train, ax=ax[0])

sns.barplot( x ='surface',y = 'avg_acceleration', data = new_x_train, palette = 'rainbow', ax=ax[1])
surfaces = new_x_train.surface.unique()

surfaces
# we need to make swarmplot for every surface showing avg velocity by group id 

for surface in surfaces:

    sns.swarmplot(x=new_x_train[new_x_train.surface == surface]['group_id'],

                  y=new_x_train[new_x_train.surface == surface]['avg_velocity'])

    plt.title('Surface = {}'.format(surface))

    plt.show()
# we need to make swarmplot for every surface showing avg velocity by group id 

for surface in surfaces:

    sns.swarmplot(x=new_x_train[new_x_train.surface == surface]['group_id'],

                  y=new_x_train[new_x_train.surface == surface]['avg_acceleration'])

    plt.title('Surface = {}'.format(surface))

    plt.show()
surfaces = new_x_train.surface.unique()

for surface in surfaces:

    sns.jointplot(x= 'avg_velocity', y = 'avg_acceleration', data =new_x_train[new_x_train.surface == surface], kind = 'reg')

    plt.title('Surface = {}'.format(surface))

    plt.show()
# pairwise correlation

x_train_2.corr(method='spearman').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
x_train_2.corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)