# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = "/kaggle/input/lhcb-jet-data-separation/"

signalData = pd.read_csv(path+"bjet_train.csv") # signal has mc_flavour = 5

backgroundData = pd.concat([pd.read_csv(path+"cjet_train.csv"), 

                            pd.read_csv(path+"ljet_train.csv")]) # background has mc_flavour != 5

print("First of {} signal rows".format(signalData.shape[0]))

display(signalData.iloc[0])

print("First of {} background rows".format(backgroundData.shape[0]))

display(backgroundData.iloc[0])
%matplotlib inline

import matplotlib.pyplot as plt

plotCols = list(signalData.columns)



for i in range(len(plotCols)):

    print("Plotting {}".format(plotCols[i]))

    plt.hist(signalData[plotCols[i]],label = "Sig")

    plt.hist(backgroundData[plotCols[i]],label = "Bkg")

    plt.legend()

    plt.xlabel(plotCols[i])

    plt.show()



# Note if the plots do not display minimise then maximise the output area below (double arrow button to the top right)
# Try fdChi2 as log10, others as linear

logCol = ['fdChi2']

linCol = ['PT', 'ETA', 'drSvrJet', 'fdrMin', 

          'm', 'mCor', 'mCorErr', 'pt', 'ptSvrJet',

          'tau', 'ipChi2Sum', 'nTrk', 'nTrkJet'] # Note skip Id as that is not helpful



from sklearn.preprocessing import MinMaxScaler,Normalizer



# redefine columns as log10(col), so ranges are more similar between variables

for l in logCol:

    signalData[l] = np.log10(signalData[l])

    backgroundData[l] = np.log10(backgroundData[l])





    #85% of the catalogue used to train, Under suggestion of Giorgio Manzoni

nTrainSig = int(signalData.shape[0]*0.85) #//2 # half the rows for training, half for evaluation

nTrainBkg = int(backgroundData.shape[0]*0.85) #//int(8./10.)#//2



# first half as training

x_data = np.concatenate([signalData[logCol+linCol][:nTrainSig].values,

                         backgroundData[logCol+linCol][:nTrainBkg].values])

y_data = np.concatenate([(signalData["mc_flavour"][:nTrainSig]==5).values.astype(np.int),

                         (backgroundData["mc_flavour"][:nTrainBkg]==5).values.astype(np.int)])



# normalized the data, under suggestion of Diego Baron

scaler = Normalizer()

scaler.fit(x_data)

scaler.transform(x_data)



nTrainSig = signalData.shape[0]//5 # half the rows for training, half for evaluation

nTrainBkg = backgroundData.shape[0]//5



#second half as evaulation

x_eval = np.concatenate([signalData[logCol+linCol][nTrainSig:].values,

                         backgroundData[logCol+linCol][nTrainBkg:].values])

y_eval = np.concatenate([(signalData["mc_flavour"][nTrainSig:]==5).values.astype(np.int),

                         (backgroundData["mc_flavour"][nTrainBkg:]==5).values.astype(np.int)])



scaler.fit(x_eval)

scaler.transform(x_eval)
# Simple 2 layer Keras network:

# import Keras overall

import keras

# a single NN layer of type "Dense" i.e. all inputs connected to all outputs

from keras.layers import Dense

# The input layer, takes x and starts NN processing

from keras.layers import Input

# Keras functional methods for defining a NN model

from keras.models import Model





# optimiser to use

# Learning rate below 1e-3, Alberto Acuto suggestion

Adam = keras.optimizers.Adam(learning_rate=0.0001) # defaults for optimiser



# define a Functional keras NN model

# define input layer

nVal = x_data.shape[1]

inputs = Input(shape=(nVal,)) 

# input->internal layer with nVal nodes



# multiple layer nodes, suggestion by Aravinda Perera

layer_nodes ={'0':int(nVal), '1':int(nVal), '2':int(nVal), '3':int(nVal), '4':int(nVal),'5':int(nVal),'6':int(nVal),'7':int(nVal)}

n_lay= len(layer_nodes)

inner_layers = {'0':Dense(int(nVal),activation='relu')(inputs)}

i=0

for n in range(n_lay):

    i=n+1

    inner_layers[str(i)] = Dense(int(nVal), activation='relu')(inner_layers[str(n)]) 



    # prev layer -> output (1 node)



    # usage of Kernel and Bias, Alberto Acuto suggestion

output = Dense(1, activation='sigmoid',use_bias=True,kernel_initializer="Constant",bias_initializer="Constant")(inner_layers[str(i)]) 

# a model is created from connecting all the layers from input to output

model = Model(inputs=inputs, outputs=output)

# Compiling the model sets up how to optimise it

model.compile(optimizer=Adam, 

              loss='binary_crossentropy', # define what is to be optimised

              metrics=['accuracy']) # what to store at each step



# run an evaluation before optimisation to see what the random initialisation

# gave as an output

score = model.evaluate(x_eval, y_eval, verbose=1)

print('Initial loss:', score[0])

print('Inital accuracy:', score[1]) # note random get you to ~60% accuracy if the data are 60% true

# Choose a batch size 

batchSize = 900 #4096

# Rather than run a fixed number of rounds, stop when the output stops improving

from keras.callbacks import EarlyStopping

# stop training early if after 10 iterations the result has not improved

early_stopping = EarlyStopping(monitor="loss", patience=50)

# Now run the optimisation, taking events from the generator

# Each epoch is one pass through the data, if not stopped do 20 epochs

history=model.fit(x=x_data,y=y_data,batch_size=batchSize,

                  verbose=1,

                  epochs=360,

                  shuffle=True, # needed as training is all true then all false

                  callbacks=[early_stopping])

print("Stopped after ",history.epoch[-1]," epochs")
# Now evaluate the model after training on the evaluation sample, should be better (but could be much improved still)

score = model.evaluate(x_eval, y_eval, verbose=1)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
testData = pd.read_csv(path+"competitionData.csv")

# apply log10 to columns that need it

for l in logCol:

    testData[l] = np.log10(testData[l])

x_test = testData[logCol+linCol].values

#scaler.fit(x_test)

#scaler.transform(x_test)

predMCFloat = model.predict(x_test)

# predMCFloat is a float: need to convert to an int

predMC = (predMCFloat>0.5).astype(np.int)

testData["Prediction1"] = predMC

pred = testData['Prediction1']



idp = np.where(pred ==1)[0]

print(len(testData), len(pred), len(idp), len(idp)/len(pred))

# solution to submit

display(testData[["Id","Prediction1"]]) # display 5 rows

# write to a csv file for submission

testData.to_csv("submit_new1.csv",index=False,columns=["Id","Prediction1"]) # Output a compressed csv file for submission: see /kaggle/working to the right