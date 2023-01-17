# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from random import randint

from sklearn.linear_model import LinearRegression



# ignore Deprecation Warning

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv(os.path.join(dirname, filename))
df.head()
df.info()
# Get the unique profile IDs

unique_profiles_id = df['profile_id'].unique()



print('Number of unique profiles: %i'%len(unique_profiles_id))
num_point_profile = np.zeros(len(unique_profiles_id))

for i in range(len(unique_profiles_id)):

    num_point_profile[i] = df[df['profile_id']==unique_profiles_id[i]].shape[0]



print('Profile ID with minimum number of points %i and it has %i points' %(unique_profiles_id[np.where(num_point_profile == np.amin(num_point_profile))],

                                                                 min(num_point_profile)))    

f, ax = plt.subplots(figsize=(18,5))

plt.bar(unique_profiles_id, (num_point_profile/df.shape[0])*100)

plt.xlabel('Profile ID')

plt.ylabel('Number of points compared to the \n total number of points(%)')
profile = randint(0,len(unique_profiles_id)) # Get a random index



corr=df[df['profile_id']==unique_profiles_id[profile]].drop('profile_id',axis=1).corr()

print('The correlation heatmap is for the profile ID %i and its is index %i'%(unique_profiles_id[profile], profile))



f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(corr, annot=True, linewidths=.5, fmt='.2f', mask= np.zeros_like(corr,dtype=np.bool), 

            cmap=sns.diverging_palette(100,200,as_cmap=True), square=True, ax=ax)



plt.show()
profile = randint(0,len(unique_profiles_id)) # Get a random index 

print('The profile ID shown is %i and it has %i points' %(unique_profiles_id[profile],num_point_profile[profile]))



sns.pairplot(df[df['profile_id']==unique_profiles_id[profile]][['i_q', 'torque']])
# Torque model



# Define vectors to collect the parameters and the score

model_torque_scores = np.zeros(len(unique_profiles_id))

coef_torque = np.zeros(len(unique_profiles_id))

inter_torque = np.zeros(len(unique_profiles_id))



for i in range (len(unique_profiles_id)) :



    model_torque = LinearRegression()



    X_torque_train = df[df['profile_id']==unique_profiles_id[i]][['i_q']].values

    y_torque_train = df[df['profile_id']==unique_profiles_id[i]][['torque']].values



    model_torque.fit(X_torque_train, y_torque_train)

    

    # Get parameters and the score

    coef_torque[i] = model_torque.coef_

    inter_torque[i] = model_torque.intercept_

    model_torque_scores[i] = model_torque.score(X_torque_train, y_torque_train)



# Representing the models parameters and scores for every profile    

f, ax = plt.subplots(figsize=(18,10))

plt.subplot(3,1,1)

plt.bar(unique_profiles_id, coef_torque)

plt.ylim((0.8,1.2))

plt.ylabel('Coefficient of L.R.')

plt.subplot(3,1,2)

plt.bar(unique_profiles_id, inter_torque)

plt.ylim((-0.1,0.1))

plt.ylabel('Intercept of L.R.')

plt.subplot(3,1,3)    

plt.bar(unique_profiles_id, model_torque_scores)

plt.ylim((0.95,1.05))

plt.xlabel('Profile ID')

plt.ylabel('Score of L.R.')
print('The coefficient of the torque model is : %f' %np.mean(coef_torque))

print('The intercept of the torque model is : %f' %np.mean(inter_torque))
profile = randint(0,len(unique_profiles_id)) # Get a random index 

print('The profile ID shown is %i and it has %i points' %(unique_profiles_id[profile],num_point_profile[profile]))



sns.pairplot(df[df['profile_id']==unique_profiles_id[profile]][['coolant', 'stator_yoke', 'stator_tooth', 'stator_winding']],

            markers ='+' )
# Stator Temperature



# Define vectors to collect the parameters and the score

model_stator_scores = np.zeros(len(unique_profiles_id))

coef_stator = np.zeros((len(unique_profiles_id),3))

inter_stator = np.zeros(len(unique_profiles_id))



for i in range (len(unique_profiles_id)) :



    model_stator = LinearRegression()



    X_stator_train = df[df['profile_id']==unique_profiles_id[i]][['coolant', 'stator_yoke', 'stator_tooth']].values

    y_stator_train = df[df['profile_id']==unique_profiles_id[i]][['stator_winding']].values



    model_stator.fit(X_stator_train, y_stator_train)

    

    # Get parameters and the score

    coef_stator[i,:] = model_stator.coef_

    inter_stator[i] = model_stator.intercept_

    model_stator_scores[i] = model_stator.score(X_stator_train, y_stator_train)



# Representing the models parameters and scores for every profile    

f, ax = plt.subplots(figsize=(18,10))

plt.subplot(5,1,1)

plt.bar(unique_profiles_id, coef_stator[:,0])

plt.ylabel('Coefficient 1 of \n M.L.R.')

plt.subplot(5,1,2)

plt.bar(unique_profiles_id, coef_stator[:,1])

plt.ylabel('Coefficient 2 of \n M.L.R.')

plt.subplot(5,1,3)

plt.bar(unique_profiles_id, coef_stator[:,2])

plt.ylabel('Coefficient 3 of \n M.L.R.')

plt.subplot(5,1,4)

plt.bar(unique_profiles_id, inter_stator)

plt.ylabel('Intercept of \n M.L.R.')

plt.subplot(5,1,5)

plt.bar(unique_profiles_id, model_stator_scores)

plt.ylim((0.75,1.05))

plt.xlabel('Profile ID')
# Calculation the average of the parameters and don't taking into consideration

# the outlier profile

print('The coefficient of the torque model is : %f' %np.mean(np.delete(coef_stator[:,0],16)))

print('The coefficient of the torque model is : %f' %np.mean(np.delete(coef_stator[:,1],16)))

print('The coefficient of the torque model is : %f' %np.mean(np.delete(coef_stator[:,2],16)))

print('The intercept of the torque model is : %f' %np.mean(np.delete(inter_stator,16)))