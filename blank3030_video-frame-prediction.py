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
from keras.models import Sequential

from keras.layers.convolutional import Conv3D

from keras.layers.convolutional_recurrent import ConvLSTM2D

from keras.layers.normalization import BatchNormalization

import numpy as np

import pylab as plt
seq = Sequential()

seq.add(ConvLSTM2D(filters = 40, kernel_size = (3,3), 

                   input_shape = (None, 40, 40, 1),

                   padding = 'same',

                   return_sequences = True))

seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters = 40, kernel_size = (3,3), 

                   padding = 'same', return_sequences = True))

seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters = 40, kernel_size = (3,3), 

                   padding = 'same', return_sequences = True))

seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters = 40, kernel_size = (3,3), 

                   padding = 'same', return_sequences = True))

seq.add(BatchNormalization())

seq.add(Conv3D(filters = 1, kernel_size = (3,3,3),

               activation = 'sigmoid', padding = 'same',

               data_format = 'channels_last'))
seq.compile(loss = 'binary_crossentropy', optimizer = 'adadelta')
def generate_movies(n_samples = 1200, n_frames =15):

    row = 80

    col = 80

    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype = np.float)

    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype = np.float)

    for i in range(n_samples):

        n = np.random.randint(3,8)

        

        for j in range(n):

            xstart = np.random.randint(20, 60)

            ystart = np.random.randint(20, 60)

            directionx = np.random.randint(0, 3) - 1

            directiony = np.random.randint(0, 3) - 1

            w = np.random.randint(2, 4)

            

            for t in range(n_frames):

                x_shift = xstart + directionx*t

                y_shift = ystart + directiony*t

                noisy_movies[i, t, x_shift - w: x_shift + w, y_shift - w: y_shift + w, 0] += 1

                if np.random.randint(0,2):

                    noise_f = (-1)**np.random.randint(0,2)

                    noisy_movies[i, t,

                                 x_shift - w - 1: x_shift + w + 1,

                                 y_shift - w - 1: y_shift + w + 1,

                                 0] += noise_f*0.1

                x_shift = xstart + directionx*(t+1)

                y_shift = ystart + directiony*(t+1)

                shifted_movies[i, t, x_shift - w: x_shift + w, y_shift - w: y_shift + w, 0] += 1

    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]

    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]

    noisy_movies[noisy_movies >= 1] = 1

    shifted_movies[shifted_movies >= 1] = 1

    return noisy_movies, shifted_movies
noisy_movies, shifted_movies = generate_movies(n_samples = 1200)
seq.fit(noisy_movies[:1000], shifted_movies[:1000], batch_size = 10, epochs = 50, validation_split =0.05)
which = 1004

track = noisy_movies[which][:7,::,::,::]
for j in range(16):

    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])

    new = new_pos[::, -1, ::, ::, ::]

    track = np.concatenate((track, new), axis = 0)

    
track2 = noisy_movies[which][::, ::, ::, ::]

for i in range(15):

    fig = plt.figure(figsize=(10, 5))



    ax = fig.add_subplot(121)



    if i >= 7:

        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')

    else:

        ax.text(1, 3, 'Initial trajectory', fontsize=20)



    toplot = track[i, ::, ::, 0]



    plt.imshow(toplot)

    ax = fig.add_subplot(122)

    plt.text(1, 3, 'Ground truth', fontsize=20)



    toplot = track2[i, ::, ::, 0]

    if i >= 2:

        toplot = shifted_movies[which][i - 1, ::, ::, 0]



    plt.imshow(toplot)

    plt.savefig('%i_animate.png' % (i + 1))