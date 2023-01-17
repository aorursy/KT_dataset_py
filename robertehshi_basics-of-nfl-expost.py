!pip install wandb
import wandb

from wandb.keras import WandbCallback



import numpy as np # linear algebra

import numpy.matlib

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import gc

import sys



# Any results you write to the current directory are saved as output.



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import SGD

from  keras.losses import categorical_crossentropy

from keras.regularizers import l2

import keras.backend as K

K.set_floatx('float32')



from kaggle.competitions import nflrush
def error_correcting_codes(df):

    df = df.replace('BLT', 'BAL')

    df = df.replace('HST', 'HOU')

    df = df.replace('ARZ', 'ARI')

    df = df.replace('CLV', 'CLE')

    return df

  



def organize_positions(df):

    return (df.loc[(df['PossessionTeam']==df['HomeTeamAbbr'])&(df['Team']=='away') | (df['PossessionTeam']==df['VisitorTeamAbbr'])&(df['Team']=='home')].copy().reset_index(),

      df.loc[((df['PossessionTeam']==df['HomeTeamAbbr'])&(df['Team']=='home') | (df['PossessionTeam']==df['VisitorTeamAbbr'])&(df['Team']=='away'))&(df['NflId']!=df['NflIdRusher'])].copy().reset_index(),

      df.loc[df['NflId']==df['NflIdRusher']].copy().reset_index())
def doubledown(X, doublings=1):

    np.random.seed(3)

    for w in range(doublings):

        X_dupe2 = np.concatenate((X.copy(), X.copy()), axis=0)

        for i in range(X.shape[0]):

            X_dupe2[2*i, :] = X[i, :]

            X_dupe2[2*i+1, :] = X[i, :]

        X = X_dupe2

    return X





def physics_doubledown(X, doublings, width):

    np.random.seed(3)

    for w in range(doublings):

        X_dupe = X.copy()

        X_dupe2 = np.concatenate((X.copy(), X.copy()), axis=0)

        zwinger = np.arange(11)

        np.random.shuffle(zwinger)

        for (i,j) in enumerate(zwinger):

            X_dupe[:, width*i:width*i+width] = X[:, width*j:width*j+width]

        zwinger = np.arange(10)

        np.random.shuffle(zwinger)

        for (i,j) in enumerate(zwinger):

            X_dupe[:, width*(i+11):width*(i+12)] = X[:, width*(j+11):width*(j+12)]

        for i in range(X.shape[0]):

            X_dupe2[2*i, :] = X[i, :]

            X_dupe2[2*i+1, :] = X_dupe[i, :]

        X = X_dupe2

    return X
def physics_init(df):

    way = -2*(df['PlayDirection']=='left') + 1

    theta = way*df['Dir']*np.pi/180

    df['X'] = (df['PlayDirection']=='right')*df['X'] + (df['PlayDirection']=='left')*(120 - df['X'])

    df['Sx'] = np.sin(theta)*df['S']

    df['Sy'] = np.cos(theta)*df['S']

    df['Ax'] = np.sin(theta)*df['A']

    df['Ay'] = np.cos(theta)*df['A']

    df['EquivYardLine'] = (df['PossessionTeam']==df['FieldPosition'])*(df['YardLine']+10) + (df['PossessionTeam']!=df['FieldPosition'])*(110-df['YardLine'])



    defn, off, RBs = organize_positions(df)



    defn['X'] -= RBs.loc[[i//11 for i in defn.index], 'X'].values

    defn['Y'] -= RBs.loc[[i//11 for i in defn.index], 'Y'].values

    defn = defn.loc[:, ('PlayId', 'PlayerWeight', 'Sx', 'Sy', 'Ax', 'Ay', 'X', 'Y')]

    defn.fillna(0, inplace=True)

    defn['Infl'] = defn['PlayerWeight']/(np.square(defn['X']) + np.square(defn['Y']))**0.5

    defn['AngularMomentum'] = -defn['PlayerWeight']*(defn['X']*defn['Sx'] + defn['Y']*defn['Sy'])/(np.square(defn['X']) + np.square(defn['Y']))



    off['X'] -= RBs.loc[[i//10 for i in off.index], 'X'].values

    off['Y'] -= RBs.loc[[i//10 for i in off.index], 'Y'].values

    off = off.loc[:, ('PlayId', 'PlayerWeight', 'Sx', 'Sy', 'Ax', 'Ay', 'X', 'Y')]

    off.fillna(0, inplace=True)

    off['Infl'] = off['PlayerWeight']/(np.square(off['X']) + np.square(off['Y']))**0.5

    off['AngularMomentum'] = -off['PlayerWeight']*(off['X']*off['Sx'] + off['Y']*off['Sy'])/(np.square(off['X']) + np.square(off['Y']))



    RBs['YardsBehindScrimmage'] = RBs['EquivYardLine'] - RBs['X']

    RBs['X'] = 0

    RBs['Y'] = 0

    RBs = RBs.loc[:, ('PlayId', 'YardsBehindScrimmage', 'PlayerWeight', 'Sx', 'Sy', 'Ax', 'Ay', 'X', 'Y')]

    RBs.fillna(0, inplace=True)

    

    return defn, off, RBs





def action(defn, off, RBs, timestep=0.1):

    t = 0.0

    while t<timestep:

        for X in (defn, off, RBs):

            X['X'] += X['Sx']*0.01 +X['Ax']*0.01**2/2

            X['Y'] += X['Sy']*0.01 +X['Ay']*0.01**2/2

            X['Sx'] += X['Ax']*0.01

            X['Sy'] += X['Ay']*0.01

            X['Ax'] *= 0.99

            X['Ay'] *= 0.99

        t += 0.01



        defn['X'] -= RBs.loc[[i//11 for i in defn.index], 'X'].values

        defn['Y'] -= RBs.loc[[i//11 for i in defn.index], 'Y'].values

        defn['Infl'] = defn['PlayerWeight']/(np.square(defn['X']) + np.square(defn['Y']))**0.5

        defn['AngularMomentum'] = -defn['PlayerWeight']*(defn['X']*defn['Sx'] + defn['Y']*defn['Sy'])/(np.square(defn['X']) + np.square(defn['Y']))



        off['X'] -= RBs.loc[[i//10 for i in off.index], 'X'].values

        off['Y'] -= RBs.loc[[i//10 for i in off.index], 'Y'].values

        off['Infl'] = off['PlayerWeight']/(np.square(off['X']) + np.square(off['Y']))**0.5

        off['AngularMomentum'] = -off['PlayerWeight']*(off['X']*off['Sx'] + off['Y']*off['Sy'])/(np.square(off['X']) + np.square(off['Y']))



        RBs['X'] = 0

        RBs['Y'] = 0



    return defn, off, RBs



def generate_physics(df, forward_action=0, timestep=0.1, doublings=0):

    d, o, r = physics_init(df)

    df = None



    defn = [d.copy()]

    off = [o.copy()]

    RBs = [r.copy()]



    for a in range(forward_action):

        d, o, r = action(d, o, r, timestep)

        defn.append(d.copy())

        off.append(o.copy())

        RBs.append(r.copy())

    d, o, r = None, None, None



    for X in (defn, off, RBs):

        for i in range(len(defn)):

            X[i]['Px'] = X[i]['Sx']*X[i]['PlayerWeight']

            X[i]['Py'] = X[i]['Sy']*X[i]['PlayerWeight']

            X[i]['Fx'] = X[i]['Ax']*X[i]['PlayerWeight']

            X[i]['Fy'] = X[i]['Ay']*X[i]['PlayerWeight']



    for i in range(len(defn)):

        if i==0:

            bigD = defn[i].loc[:, ('Px', 'Py', 'X', 'Y', 'Infl', 'AngularMomentum')].astype(np.float32)

            bigO = off[i].loc[:, ('Px', 'Py', 'X', 'Y', 'Infl', 'AngularMomentum')].astype(np.float32)

            backs = RBs[i].loc[:, ('YardsBehindScrimmage', 'Px', 'Py', 'Fx', 'Fy')].astype(np.float32)

            inst = np.concatenate((np.reshape(bigD.copy().values, (bigD.shape[0]//11, 11*6)), 

                                            np.reshape(bigO.copy().values, (bigO.shape[0]//10, 10*6)), 

                                            backs.copy().values), axis=1)

            summary = physics_doubledown(inst, doublings, 6)

        else:

            bigD = defn[i].loc[:, ('Infl', 'AngularMomentum')].astype(np.float32)

            bigO = off[i].loc[:, ('Infl', 'AngularMomentum')].astype(np.float32)

            inst = np.concatenate((np.reshape(bigD.copy().values, (bigD.shape[0]//11, 11*2)), 

                                            np.reshape(bigO.copy().values, (bigO.shape[0]//10, 10*2))), axis=1)

            summary = np.concatenate((summary, physics_doubledown(inst, doublings, 2)), axis=1)

    defn, off, RBs = None, None, None



    return summary
def down_situation(df):

    X = df.loc[::22, ('Down', 'Distance')].copy().astype(np.float32)

    X.fillna(-1, inplace=True)

    framer = pd.DataFrame(columns=(1.0, 2.0, 3.0, 4.0, 'Distance'), dtype=np.float32)

    concatenation = pd.concat((pd.get_dummies(X['Down']), X['Distance']), axis=1, join='outer')

    concatenation = pd.concat((framer, concatenation), axis=0, join='outer')

    concatenation.fillna(0, inplace=True)

    return concatenation.values
def stats(array):

    return array.mean(axis=0), array.std(axis=0)





def normalize(array, mn, stand):

    return (array - mn) / stand
def generate_yardudge(df):

    Y = np.zeros((df.shape[0]//22, 199))

    for (i, yerds) in enumerate(df['Yards'][::22]):

        Y[i, yerds+99] = 1

    return Y.astype(np.float32)
def chirps(y_true, y_pred):

    Y = np.reshape(np.argmax(y_true, axis=1) - 99, (-1,1))

    P = np.zeros((y_pred.shape[0], y_pred.shape[1]))

    stuff = np.zeros(y_pred.shape[0])

    for i in range(199):

        stuff += y_pred[:, i]

        P[:,i] = stuff

    H = (np.reshape(np.array(range(-99,100)), (1, 199)) - Y) >= 0

    CRPS = K.square(P - H)

    return K.mean(CRPS)
global meen, sigma, PCs, doublings, forward_actions, timestep



train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)



doublings = 6

forward_actions = 8

timestep = 0.1



Y = doubledown(generate_yardudge(train_df), doublings)



train_df = error_correcting_codes(train_df)

X = np.concatenate(

                (doubledown(down_situation(train_df), doublings), 

                 generate_physics(train_df, forward_actions, timestep, doublings)), 

            axis=1)

train_df = None



meen, sigma = stats(X)

X = normalize(X, meen, sigma)



PCs = np.linalg.eig(np.dot(np.transpose(X), X))[1]

X = np.dot(X, PCs)
Xtrain = X[0:int(7/8*X.shape[0]), :]

Ytrain = Y[0:int(7/8*Y.shape[0]), :]

Xtest = X[int(7/8*X.shape[0]):, :]

Ytest = Y[int(7/8*Y.shape[0]):, :]
wandb.init(anonymous='allow', project="kaggle")



wandb.config.learning_rate = 0.01541

wandb.config.decay = 0.00000173

wandb.config.beta = 0.8858

wandb.config.clipse = 0.3867

wandb.config.dropout_rate = 0.6044

wandb.config.epochs = 14

wandb.config.batch_size = 180

wandb.config.hidden_layer1_size = 1024

wandb.config.hidden_layer2_size = 512

wandb.config.hidden_layer3_size = 256



np.random.seed(1729)

model = Sequential()

model.add(Dense(units=wandb.config.hidden_layer1_size, activation='relu'))

model.add(Dropout(wandb.config.dropout_rate))

model.add(Dense(units=wandb.config.hidden_layer2_size, activation='relu'))

model.add(Dropout(wandb.config.dropout_rate))

model.add(Dense(units=wandb.config.hidden_layer3_size, activation='relu'))

model.add(Dropout(wandb.config.dropout_rate))

model.add(Dense(units=199, activation='softmax'))

model.compile(loss=categorical_crossentropy, optimizer=SGD(lr=wandb.config.learning_rate, decay=wandb.config.decay, 

                                                                       momentum=wandb.config.beta, nesterov=True, clipnorm=wandb.config.clipse))

%%wandb



# for epoch in range(wandb.config.epochs):

#     hist = model.fit(Xtrain, Ytrain, epochs=1, batch_size=wandb.config.batch_size, validation_data=(Xtest, Ytest))

#     Ptrain = model.predict(Xtrain)

#     Ptest = model.predict(Xtest)

#     loss = hist.history['loss']

#     val_loss = hist.history['val_loss']

#     CRPS = chirps(Ytrain, Ptrain)

#     val_CRPS = chirps(Ytest, Ptest)

#     wandb.log({'epoch': epoch, 'loss': loss, 'val_loss':val_loss, 'CRPS':CRPS, 'val_CRPS':val_CRPS}, step=epoch)

        

        

        

model.fit(X, Y, epochs=wandb.config.epochs, batch_size=wandb.config.batch_size, validation_split=0.1, callbacks=[WandbCallback()])