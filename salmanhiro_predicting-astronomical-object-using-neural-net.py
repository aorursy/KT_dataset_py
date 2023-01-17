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
df = pd.read_csv('/kaggle/input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
df.head()
df.columns
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
df['class'].value_counts()
from astropy import units as u

from astropy.coordinates import SkyCoord

from astropy.cosmology import WMAP9 as cosmo



radec = SkyCoord(ra=df['ra']*u.degree, dec=df['dec']*u.degree, frame='icrs')

#radec.ra.value

#radec.dec.value

galactic = radec.galactic



df['l'] = galactic.l.value

df['b'] = galactic.b.value





r = cosmo.comoving_distance(df['redshift'])

df['distance']= r.value



df.head()
def cartesian(dist,alpha,delta):

    x = dist*np.cos(np.deg2rad(delta))*np.cos(np.deg2rad(alpha))

    y = dist*np.cos(np.deg2rad(delta))*np.sin(np.deg2rad(alpha))

    z = dist*np.sin(np.deg2rad(delta))

    return x,y,z



cart = cartesian(df['distance'],df['ra'],df['dec'])

df['x_coord'] = cart[0]

df['y_coord'] = cart[1]

df['z_coord'] = cart[2]



df.head()
df['u-r'] = df['u']-df['r']
galaxy = df[df['class']=='GALAXY']

star = df[df['class']=='STAR']

quasar = df[df['class']=='QSO']
fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(galaxy['ra'],galaxy['dec'],galaxy['redshift'], s = 0.7, color = 'b', label = 'galaxy')

ax.scatter(star['ra'],star['dec'],star['redshift'], s = 0.7, color = 'y', label = 'star')

ax.scatter(quasar['ra'],quasar['dec'],quasar['redshift'], s = 0.7, color = 'r', label = 'quasar')

ax.set_xlabel('ra')

ax.set_ylabel('dec')

ax.set_zlabel('z')

ax.set_title('Object Distribution from SDSS',fontsize=18)

plt.legend()

plt.show()
plt.scatter(galaxy['r'], galaxy['u-r'], s = 0.9, color = 'b')

plt.scatter(star['r'], star['u-r'], s = 0.9, color = 'y')

plt.scatter(quasar['r'], quasar['u-r'], s = 0.9, color = 'r')

plt.xlabel('r')

plt.ylabel('u-r')

plt.title('CMD')

galaxy.head()
fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(galaxy['ra'],galaxy['dec'],galaxy['redshift'], s = 0.7)

ax.set_xlabel('ra')

ax.set_ylabel('dec')

ax.set_zlabel('z')

ax.set_title('Galactic Distribution from SDSS',fontsize=18)

plt.show()
fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(galaxy['x_coord'],galaxy['y_coord'],galaxy['z_coord'], s = 0.7, color = 'b')

ax.set_xlabel('x_coord')

ax.set_ylabel('y_coord')

ax.set_zlabel('z_coord')

ax.set_title('Galoaxy Distribution from SDSS',fontsize=18)

plt.show()



# z is the position from galaxy in cartesian coordinate, not to be confused with redshift
sns.distplot(galaxy['redshift'], kde = False)
sns.distplot(galaxy['distance'], kde = False)

plt.title('Distance (Mpc)')
star.head()
fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(star['ra'],star['dec'],star['redshift'], s = 0.7, color = 'y')

ax.set_xlabel('ra')

ax.set_ylabel('dec')

ax.set_zlabel('z')

ax.set_title('Star Distribution from SDSS',fontsize=18)

plt.show()
fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(star['x_coord'],star['y_coord'],star['z_coord'], s = 0.7, color = 'y')

ax.set_xlabel('x_coord')

ax.set_ylabel('y_coord')

ax.set_zlabel('z_coord')

ax.set_title('Distribution from SDSS',fontsize=18)

plt.show()



# z is the position from galaxy in cartesian coordinate, not to be confused with redshift
sns.distplot(star['redshift'], kde = False)

quasar.head()
fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(quasar['ra'],quasar['dec'],quasar['redshift'], s = 0.7, color = 'r')

ax.set_xlabel('ra')

ax.set_ylabel('dec')

ax.set_zlabel('z')

ax.set_title('QSO Distribution from SDSS',fontsize=18)

plt.show()
fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(quasar['x_coord'],quasar['y_coord'],quasar['z_coord'], s = 0.7, color = 'r')

ax.set_xlabel('x_coord')

ax.set_ylabel('y_coord')

ax.set_zlabel('z_coord')

ax.set_title('Distribution from SDSS',fontsize=18)

plt.show()



# z is the position from galaxy in cartesian coordinate, not to be confused with redshift
df.head()
fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(galaxy['x_coord'],galaxy['y_coord'],galaxy['z_coord'], s = 0.7, color = 'b')

ax.scatter(star['x_coord'],star['y_coord'],star['z_coord'], s = 0.7, color = 'y')

ax.scatter(quasar['x_coord'],quasar['y_coord'],quasar['z_coord'], s = 0.7, color = 'r')

ax.set_xlabel('x_coord')

ax.set_ylabel('y_coord')

ax.set_zlabel('z_coord')

ax.set_title('Distribution from SDSS',fontsize=18)

plt.show()



# z is the position from galaxy in cartesian coordinate, not to be confused with redshift
display(df.head())

display(df.columns)
df['class'] = df['class'].astype('category').cat.codes
df['class'].value_counts()
df.columns
X_df = df.drop(['objid','class'], axis=1).values

y_df = df['class'].values
display(X_df)

display(y_df)
from sklearn.preprocessing import StandardScaler, MinMaxScaler



#ss = StandardScaler()

#X_df = ss.fit_transform(X_df)

minmax = MinMaxScaler()

X_df = minmax.fit_transform(X_df)
y_df = y_df.reshape(-1,1)
display(X_df)

display(y_df)
from sklearn.preprocessing import OneHotEncoder



enc = OneHotEncoder()

y_df = enc.fit_transform(y_df).toarray()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_df,y_df, test_size=0.25)

X_train.shape
y_train.shape
import keras

from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

model.add(Dense(64, input_dim=23, activation='sigmoid'))

model.add(Dense(32, activation='sigmoid'))

model.add(Dense(16, activation='sigmoid'))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split = 0.1, epochs=30, batch_size=32)

prediction = []

test = []
y_predict = model.predict(X_test)
for i in range(len(y_test)): 

    prediction.append(np.argmax(y_predict[i]))

    test.append(np.argmax(y_test[i])) 
from sklearn.metrics import accuracy_score

acc = accuracy_score(prediction,test) 

print('Accuracy is:', acc*100, '%')
compare = pd.DataFrame(prediction, columns = ['prediction'])

compare['test'] = test
result = pd.DataFrame(X_test, columns = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'run', 'rerun', 'camcol',

       'field', 'specobjid', 'redshift', 'plate', 'mjd', 'fiberid',

       'l', 'b', 'distance', 'x_coord', 'y_coord', 'z_coord', 'u-r'])
result['class'] = compare['test']

result['prediction'] = compare['prediction']



result.to_csv('object_prediction.csv', index = False)