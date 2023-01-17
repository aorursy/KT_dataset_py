import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

#import seaborn library for plots

import seaborn as sns

%matplotlib inline
#read in data and verify

names = ['Rank','Name', 'Platform', 'Year', 'Genre','Publisher','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']

df = pd.read_csv('../input/vgsales2.csv', header=None,delim_whitespace=False,names=names,na_values='?')

df.head(6)
df.shape
#drop all rows with empty slots

df = df.dropna()
df.shape
df.head(6)
#drop all rows past 2015

df = df[df.Year < 2016]

df.shape
df.head(6)
#display the values of the Publishers so that we can get a top 10

df.Publisher.value_counts()
#drop all rows that aren't in the top 10

dfP1 = df[df.Publisher == 'Electronic Arts']

dfP2 = df[df.Publisher == 'Activision']

dfP3 = df[df.Publisher == 'Ubisoft']

dfP4 = df[df.Publisher == 'Namco Bandai Games']

dfP5 = df[df.Publisher == 'Konami Digital Entertainment']

dfP6 = df[df.Publisher == 'THQ']

dfP7 = df[df.Publisher == 'Nintendo']

dfP8 = df[df.Publisher == 'Sony Computer Entertainment']

dfP9 = df[df.Publisher == 'Sega']

dfP10 = df[df.Publisher == 'Take-Two Interactive']

df = pd.concat([dfP1,dfP2,dfP3,dfP4,dfP5,dfP6,dfP7,dfP8,dfP9,dfP10])

df = df.sort_values('Rank',ascending=True)

df.head(6)
df.shape
dfXB = df[df.Platform == 'XB']
dfXB.shape
dfXB.head(6)
dfX360 = df[df.Platform == 'X360']

dfX360.shape
dfX360.head(6)
dfPC = df[df.Platform == 'PC']

dfPC.shape
dfPC.head(6)
dfPS = df[df.Platform == 'PS']

dfPS.shape
dfPS.head(6)
dfPS2 = df[df.Platform == 'PS2']

dfPS2.shape
dfPS2.head(6)
dfPS3 = df[df.Platform == 'PS3']

dfPS3.shape
dfPS3.head(6)
#Group platforms and reorder

df2 = pd.concat([dfXB,dfX360,dfPC,dfPS,dfPS2,dfPS3])

df2.shape
df2 = df2.sort_values('Rank',ascending=True)

df2.head(6)
#Label Encoding

from sklearn import linear_model, preprocessing

le = preprocessing.LabelEncoder()

df3 = df2.apply(le.fit_transform)

df3.head(6)
platform = np.array(df3['Platform'])

genre = np.array(df3['Genre'])

publisher = np.array(df3['Publisher'])

US = np.array(df3['NA_Sales'])

EU = np.array(df3['EU_Sales'])

JP = np.array(df3['JP_Sales'])

Global = np.array(df3['Global_Sales'])

X = np.vstack((platform, genre,publisher))

X = X.T
nsamp, natt = X.shape
y = US

ym = np.mean(y)

syy = np.mean((y-ym)**2)

Rsq = np.zeros(natt)

beta0 = np.zeros(natt)

beta1 = np.zeros(natt)

for k in range(natt):

    xm = np.mean(X[:,k])

    sxy = np.mean((X[:,k]-xm)*(y-ym))

    sxx = np.mean((X[:,k]-xm)**2)

    beta1[k] = sxy/sxx

    beta0[k] = ym - beta1[k]*xm

    Rsq[k] = (sxy)**2/sxx/syy

    

    print("{0:2d}  Rsq={1:f}".format(k,Rsq[k]))
ns_train = nsamp // 2

ns_test = nsamp - ns_train

X_tr = X[:ns_train,:]

y_tr = y[:ns_train]

regr = linear_model.LinearRegression()

regr.fit(X_tr,y_tr)
y_tr_pred = regr.predict(X_tr)

RSS_tr = np.mean((y_tr_pred-y_tr)**2)/(np.std(y_tr)**2)

Rsq_tr = 1-RSS_tr

print("RSS per sample = {0:f}".format(RSS_tr))

print("R^2 =            {0:f}".format(Rsq_tr))
X_test = X[ns_train:,:]

y_test = y[ns_train:]

y_test_pred = regr.predict(X_test)

RSS_test = np.mean((y_test_pred-y_test)**2)/(np.std(y_test)**2)

Rsq_test = 1-RSS_test

print("RSS per sample = {0:f}".format(RSS_test))

print("R^2 =            {0:f}".format(Rsq_test))
import keras

from keras import optimizers

from keras.models import Model, Sequential

from keras.layers import Dense, Activation
import keras.backend as K

K.clear_session()
nin = 3 #number of inputs

nh = 4 #number of hidden layers

nout = 1 #number of outputs

model = Sequential()

model.add(Dense(nh, input_shape=(nin,), activation='sigmoid', name='hidden'))

model.add(Dense(1, activation='sigmoid', name='output'))

model.summary()
#train the network

opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
nit = 20   # number of training iterations

nepoch_per_it = 50  # number of epochs per iterations



# Loss, accuracy and epoch per iteration

loss = np.zeros(nit)

acc = np.zeros(nit)

epoch_it = np.zeros(nit)



# Main iteration loop

for it in range(nit):

    

    # Continue the fit of the model

    init_epoch = it*nepoch_per_it

    model.fit(X, y, epochs=nepoch_per_it, batch_size=100, verbose=0)

    

    # Measure the loss and accuracy on the training data

    lossi, acci = model.evaluate(X,y, verbose=0)

    epochi = (it+1)*nepoch_per_it

    epoch_it[it] = epochi

    loss[it] = lossi

    acc[it] = acci

    print("epoch=%4d loss=%12.4e acc=%7.5f" % (epochi,lossi,acci))
y = Global

ym = np.mean(y)

syy = np.mean((y-ym)**2)

Rsq = np.zeros(natt)

beta0 = np.zeros(natt)

beta1 = np.zeros(natt)

for k in range(natt):

    xm = np.mean(X[:,k])

    sxy = np.mean((X[:,k]-xm)*(y-ym))

    sxx = np.mean((X[:,k]-xm)**2)

    beta1[k] = sxy/sxx

    beta0[k] = ym - beta1[k]*xm

    Rsq[k] = (sxy)**2/sxx/syy

    

    print("{0:2d}  Rsq={1:f}".format(k,Rsq[k]))
ns_train = nsamp // 2

ns_test = nsamp - ns_train

X_tr = X[:ns_train,:]

y_tr = y[:ns_train]

regr = linear_model.LinearRegression()

regr.fit(X_tr,y_tr)
y_tr_pred = regr.predict(X_tr)

RSS_tr = np.mean((y_tr_pred-y_tr)**2)/(np.std(y_tr)**2)

Rsq_tr = 1-RSS_tr

print("RSS per sample = {0:f}".format(RSS_tr))

print("R^2 =            {0:f}".format(Rsq_tr))
K.clear_session()
nin = 3 #number of inputs

nh = 4 #number of hidden layers

nout = 1 #number of outputs

model = Sequential()

model.add(Dense(nh, input_shape=(nin,), activation='sigmoid', name='hidden'))

model.add(Dense(1, activation='sigmoid', name='output'))

model.summary()
#train the network

opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
nit = 20   # number of training iterations

nepoch_per_it = 50  # number of epochs per iterations



# Loss, accuracy and epoch per iteration

loss = np.zeros(nit)

acc = np.zeros(nit)

epoch_it = np.zeros(nit)



# Main iteration loop

for it in range(nit):

    

    # Continue the fit of the model

    init_epoch = it*nepoch_per_it

    model.fit(X, y, epochs=nepoch_per_it, batch_size=100, verbose=0)

    

    # Measure the loss and accuracy on the training data

    lossi, acci = model.evaluate(X,y, verbose=0)

    epochi = (it+1)*nepoch_per_it

    epoch_it[it] = epochi

    loss[it] = lossi

    acc[it] = acci

    print("epoch=%4d loss=%12.4e acc=%7.5f" % (epochi,lossi,acci))