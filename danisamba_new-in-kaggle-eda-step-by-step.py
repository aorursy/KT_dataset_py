import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

#for displaying images

from IPython.display import Image
# Table printing large

plt.rcParams['figure.figsize'] = (15, 7)

pd.set_option("display.max_columns", 400)

pd.options.display.max_colwidth = 250

pd.set_option("display.max_rows", 100)

# High defition plots

%config InlineBackend.figure_format = 'retina'

sns.set()
df_train = pd.read_csv('../input/career-con-2019/X_train.csv')

y_train = pd.read_csv('../input/career-con-2019/y_train.csv')



print('X_train.csv shape is {}'.format(df_train.shape))

print('y_train.csv shape is {}'.format(y_train.shape))
Image("../input/careercon2019/robot.JPG",width=400)
Image("../input/careercon2019/IMU.png",width=400)
Image("../input/careercon2019/vector.jpg",width=400)
df_train.head()
df_train.shape[0]/y_train.shape[0]
y_train.shape[0]
y_train.head(5)
print('Number of classes: {}'.format(y_train.surface.nunique()))

print('Number of group_id: {}'.format(y_train.group_id.nunique()))
sns.catplot(x='surface',data=y_train,kind='count')

plt.xticks(rotation=90)

plt.show()

print(y_train.surface.value_counts(normalize=True))
sns.countplot(x='group_id',data=y_train)

plt.xticks(rotation=90)

plt.show()
y_train.groupby('group_id').surface.nunique().max()
#Function to calculate the norm of a three element vector

def vector_norm(x,y,z,df):

    return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)
df_train['angular_velocity_norm'] =vector_norm('angular_velocity_X',

                                                'angular_velocity_Y',

                                                'angular_velocity_Z',df_train)



df_train['linear_acceleration_norm'] =vector_norm('linear_acceleration_X',

                                                'linear_acceleration_Y',

                                                'linear_acceleration_Z',df_train)
new_df = df_train.groupby('series_id')['angular_velocity_norm','linear_acceleration_norm'].mean()

new_df = pd.DataFrame(new_df).reset_index()

new_df.columns = ['serie_id','avg_velocity','avg_acceleration']

new_df['surface'] = y_train.surface

new_df['group_id'] = y_train.group_id
new_df.head(3)
sns.boxplot(x='surface',y='avg_velocity',data=new_df)

plt.title('avg_velocity vs surface')
surfaces = new_df.surface.unique()



for surface in surfaces:

    sns.swarmplot(x=new_df[new_df.surface == surface]['group_id'],

                  y=new_df[new_df.surface == surface]['avg_velocity'])

    plt.title('Surface = {}'.format(surface))

    plt.show()
sns.boxplot(x='surface',y='avg_acceleration',data=new_df)

plt.title('Avg_acceleration vs Surface')
for surface in surfaces:

    sns.swarmplot(x=new_df[new_df.surface == surface]['group_id'],

                  y=new_df[new_df.surface == surface]['avg_acceleration'])

    plt.title('Surface = {}'.format(surface))

    plt.show()