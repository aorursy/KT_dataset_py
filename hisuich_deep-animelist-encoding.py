# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
userAnimeListDataPd=pd.read_csv('/kaggle/input/myanimelist/UserAnimeList.csv',

                              usecols=['username', 'anime_id', 'my_score'],

                               dtype={'username':'string_', 'anime_id':'uint16', 'my_score':'uint8'},

                              #nrows=1000 #for testing

                               )

userAnimeListDataPd=userAnimeListDataPd.dropna()

userAnimeListDataPd=userAnimeListDataPd[userAnimeListDataPd['my_score']>7.0]

#trim

nrows=100000

userAnimeListDataPd=userAnimeListDataPd.iloc[0:nrows]



userAnimeListDataPd.head(20)
from sklearn.preprocessing import LabelEncoder



#userAnimeListDataPd=userAnimeListDataPd.set_index('username')

labelencoder = LabelEncoder()

userAnimeListDataPd['username'] = labelencoder.fit_transform(userAnimeListDataPd['username'])

userAnimeListDataPd.head()
print('Shape: ',userAnimeListDataPd.shape)

number_of_users=userAnimeListDataPd.drop_duplicates(subset=['username']).shape[0]

print('Number of users: ',number_of_users)

titles=userAnimeListDataPd.drop_duplicates(subset=['anime_id']).shape[0]

print("number of titles: ", titles)

#userAnimeListDataPd=userAnimeListDataPd[['username', 'anime_id']].set_index('username').stack()

#userAnimeListDataPd=pd.get_dummies(userAnimeListDataPd, sparse=True).groupby(level=0).sum().head()

#titles=userAnimeListDataPd.shape[1]
#userAnimeListDataPd.index.name='username'

userAnimeListDataPd =userAnimeListDataPd[['username']].join(pd.get_dummies(userAnimeListDataPd['anime_id'], sparse=True)).groupby('username').max()

#userAnimeListDataPd =userAnimeListDataPd['index'].join(pd.get_dummies(userAnimeListDataPd['anime_id'])).groupby('index').max()

#userAnimeListDataPd.pivot_table(index=['number_label'], columns=['anime_id'], aggfunc=[len], fill_value=0)

print(userAnimeListDataPd.head(20))

print(userAnimeListDataPd.shape)
userAnimeListDataPd.to_csv (r'/kaggle/exported.csv', header=True, compression='zip')
from keras.layers import Input, Dense, Flatten, Reshape, Concatenate

from keras import models



class deep_autoencoder():

    #for 2d graph

    

    def __init__(self,input_size, l1, l2):    

        "https://blog.keras.io/building-autoencoders-in-keras.html"

        

        encoded_dim=2

        input_shape=Input(shape=(input_size,))

        #encoder

        #self.encoder=models.Sequential()

        self.encoder=Dense(l1, activation='relu')(input_shape)

        self.encoder=Dense(l2, activation='tanh')(self.encoder)

        self.encoder=Dense(encoded_dim, activation='tanh')(self.encoder)



        #decoder

        

        #self.decoder=models.Sequential()

        self.decoder=Dense(l2, activation='tanh', input_dim=encoded_dim)(self.encoder)

        self.decoder=Dense(l1, activation='tanh')(self.decoder)

        self.decoder=Dense(input_size, activation='relu')(self.decoder)

        

        #autoencoder

        #self.model=models.Model(input_shape, self.encoder(self.decoder(input_shape)), name="autoencoder")

        self.model=models.Model(input_shape,self.decoder)

        self.encmodel=models.Model(input_shape,self.encoder)

    

    
autoencoder=deep_autoencoder(titles, 900, 100)

autoencoder.model.compile(loss='mse', optimizer='adam')

autoencoder.model.fit(userAnimeListDataPd, userAnimeListDataPd, epochs=20)
import matplotlib.pyplot as plt

z=autoencoder.encmodel.predict(userAnimeListDataPd)

print(userAnimeListDataPd.shape)

plt.scatter(z[:,0], z[:,1], marker='o', s=0.1, c='#d53a26')

plt.show()

plt.savefig('stat.png', dpi=600)

fig, axs=plt.subplots(19, figsize=(10, 190))

for i in range(1,20):

    autoencoder=deep_autoencoder(titles, 5*i, 3*i)

    autoencoder.model.compile(loss='mse', optimizer='adam')

    autoencoder.model.fit(userAnimeListDataPd, userAnimeListDataPd, epochs=20)

    import matplotlib.pyplot as plt

    z=autoencoder.encmodel.predict(userAnimeListDataPd)

    print(userAnimeListDataPd.shape)

    #axs[i-1].set_title("NN size = ", str(10*i), str(3*i))

    axs[i-1].scatter(z[:,0], z[:,1], marker='o', s=0.1, c='#d53a26')

autoencoder=deep_autoencoder(titles, 6, 3)

autoencoder.model.compile(loss='mse', optimizer='adam')

autoencoder.model.fit(userAnimeListDataPd, userAnimeListDataPd, epochs=120)

z=autoencoder.encmodel.predict(userAnimeListDataPd)

print(userAnimeListDataPd.shape)

#axs[i-1].set_title("NN size = ", str(10*i), str(3*i))

plt.scatter(z[:,0], z[:,1], marker='o', s=0.1, c='#d53a26')
model_json = autoencoder.model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

model_json = autoencoder.encmodel.to_json()

with open("encmodel.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

autoencoder.model.save_weights("encmodel.h5")

print("Saved model to disk")