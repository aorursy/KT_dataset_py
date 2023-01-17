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
# Load require libraries

# Plotting
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

# Neural Network
dataTrainRaw = pd.read_csv( "../input/train.csv" )
dataTestRaw  = pd.read_csv( "../input/test.csv" )

dataTrainFeatRaw = (dataTrainRaw.iloc[:,1:].values).astype( 'float32' )
dataTrainLabels  = dataTrainRaw.iloc[:,0].values.astype('int32')
dataTestFeatRaw  = dataTestRaw.values.astype('float32')

print( "Train Features : ", dataTrainFeatRaw.shape )
print( "Train Labels   : ", dataTrainLabels.shape )
print( "Test Features  : ", dataTestFeatRaw.shape )
from sklearn.model_selection import train_test_split

dataCvTrainFeatRaw, dataCvTestFeatRaw, dataCvTrainLabels, dataCvTestLabels = train_test_split( dataTrainFeatRaw, dataTrainLabels, test_size = 0.10 )

print( "Cross Validation: " )
print( "    Train Features : ", dataCvTrainFeatRaw.shape )
print( "    Train Labels   : ", dataCvTrainLabels.shape )
print( "    Test Features  : ", dataCvTestFeatRaw.shape )
print( "    Test Labels    : ", dataCvTestLabels.shape )

from keras.utils import to_categorical
dataCvTrainOnehot = to_categorical( dataCvTrainLabels, num_classes = 10 )
dataCvTestOnehot  = to_categorical( dataCvTestLabels, num_classes = 10 )
dataTrainOnehot   = to_categorical( dataTrainLabels, num_classes = 10 )
# Feature Engineering
from sklearn.preprocessing import StandardScaler

dataStandardizer = StandardScaler()
dataStandardizer.fit( dataTrainFeatRaw )

dataCvTrainFeatStd = dataStandardizer.transform( dataCvTrainFeatRaw )
dataCvTestFeatStd  = dataStandardizer.transform( dataCvTestFeatRaw )

print( "Before standardization: " )
print( "    Train:        Mean = {:+9.6f}    SD = {:9.6f}".format(np.mean(dataCvTrainFeatRaw),np.std(dataCvTrainFeatRaw)) )
print( "    Test :        Mean = {:+9.6f}    SD = {:9.6f}".format(np.mean(dataCvTestFeatRaw),np.std(dataCvTestFeatRaw)) )
print( "After standardization: " )
print( "    Train:        Mean = {:+9.6f}    SD = {:9.6f}".format(np.mean(dataCvTrainFeatStd),np.std(dataCvTrainFeatStd)) )
print( "    Test :        Mean = {:+9.6f}    SD = {:9.6f}".format(np.mean(dataCvTestFeatStd),np.std(dataCvTestFeatStd)) )
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam ,RMSprop

# Utility function to create models
def make_model( inDim, outDim, hDims = [] ):
    model = Sequential()
    if not hDims:
        model.add( Dense( outDim, input_dim = inDim, activation = 'softmax') )
    else:
        model.add( Dense( hDims[0], input_dim = inDim, activation = "relu" ) )
        for i in range(1,len(hDims)):
            model.add( Dense( hDims[i], activation = "relu" ) )
        model.add( Dense( outDim, activation = "softmax") )
    model.compile( optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"] )
    
    return model

# For training models and ploting results
def train_model( model, trainX, trainY, valX, valY, epochs = 50, verbose = 2, batch_size = 32 ):
#     print( "Train    X: ", trainX.shape, "  Y: ", trainY.shape )
#     print( "Train    X: ", valX.shape, "  Y: ", valY.shape )
    valData = (valX,valY)
    if valX is None:
        valData = None
    trainInfo = model.fit( trainX, trainY, validation_data = valData, epochs = epochs, verbose = verbose, batch_size = batch_size )
    
    # Plot Results
    trainHistory = trainInfo.history
    trainHistory.keys()

    fig, ax = plt.subplots( 1, 2, figsize = (12,4) )

    ax[0].plot( trainHistory['loss'], label = "Train" )
    if valX is not None:
        ax[0].plot( trainHistory['val_loss'], label = "Validation" )
    ax[0].set_title( "Loss" )
    ax[0].set_xlabel( "Epochs" )
    ax[0].set_ylabel( "Loss" )
    ax[0].grid( True )
    ax[0].legend()

    ax[1].plot( trainHistory['acc'], label = "Train" )
    if valX is not None:
        ax[1].plot( trainHistory['val_acc'], label = "Validation" )
    ax[1].set_title( "Accuracy" )
    ax[1].set_xlabel( "Epochs" )
    ax[1].set_ylabel( "Accuracy" )
    ax[1].grid( True )
    ax[1].legend()
    
    return trainInfo
baselineModel1 = make_model( dataCvTrainFeatRaw.shape[1], 10, [32,32] )
train_model( baselineModel1, dataCvTrainFeatRaw, dataCvTrainOnehot, dataCvTestFeatRaw, dataCvTestOnehot )
baselineModel2 = make_model( dataCvTrainFeatRaw.shape[1], 10, [64,64,32,32] )
train_model( baselineModel2, dataCvTrainFeatRaw, dataCvTrainOnehot, dataCvTestFeatRaw, dataCvTestOnehot )
normModel1 = make_model( dataCvTrainFeatStd.shape[1], 10, [32,32] )
train_model( normModel1, dataCvTrainFeatStd, dataCvTrainOnehot, dataCvTestFeatStd, dataCvTestOnehot )
normModel2 = make_model( dataCvTrainFeatStd.shape[1], 10, [64,64,32,32] )
train_model( normModel2, dataCvTrainFeatStd, dataCvTrainOnehot, dataCvTestFeatStd, dataCvTestOnehot )
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

lda = LDA()
lda.fit( dataCvTrainFeatStd, dataCvTrainLabels )

dataCvTrainFeatLDA = lda.transform( dataCvTrainFeatStd )
dataCvTestFeatLDA  = lda.transform( dataCvTestFeatStd )

print( "Shape after LDA:    Train = ", dataCvTrainFeatLDA.shape, "    Test = ", dataCvTestFeatLDA.shape )
ldaModel1 = make_model( dataCvTrainFeatLDA.shape[1], 10, [32,32] )
train_model( ldaModel1, dataCvTrainFeatLDA, dataCvTrainOnehot, dataCvTestFeatLDA, dataCvTestOnehot )
ldaModel2 = make_model( dataCvTrainFeatLDA.shape[1], 10, [64,64,32,32] )
train_model( ldaModel2, dataCvTrainFeatLDA, dataCvTrainOnehot, dataCvTestFeatLDA, dataCvTestOnehot )
from sklearn.metrics import accuracy_score

acc_baselineModel1 = accuracy_score( dataCvTestLabels, np.argmax(baselineModel1.predict(dataCvTestFeatRaw),1) );
acc_baselineModel2 = accuracy_score( dataCvTestLabels, np.argmax(baselineModel2.predict(dataCvTestFeatRaw),1) );

acc_normModel1     = accuracy_score( dataCvTestLabels, np.argmax(normModel1.predict(dataCvTestFeatStd),1) );
acc_normModel2     = accuracy_score( dataCvTestLabels, np.argmax(normModel2.predict(dataCvTestFeatStd),1) );

acc_ldaModel1      = accuracy_score( dataCvTestLabels, np.argmax(ldaModel1.predict(dataCvTestFeatLDA),1) );
acc_ldaModel2      = accuracy_score( dataCvTestLabels, np.argmax(ldaModel2.predict(dataCvTestFeatLDA),1) );

acc_ldaPlain       = accuracy_score( dataCvTestLabels, lda.predict(dataCvTestFeatStd) )


print( "Results" )
print( "Baseline Models: " )
print( "    Small Network    = ", acc_baselineModel1 )
print( "    Large Network    = ", acc_baselineModel2 )
print( "Models with data normalization: " )
print( "    Small Network    = ", acc_normModel1 )
print( "    Large Network    = ", acc_normModel2 )
print( "LDA based Models: " )
print( "    Small Network    = ", acc_ldaModel1 )
print( "    Large Network    = ", acc_ldaModel2 )
print( "Plain LDA            = ", acc_ldaPlain )


data = [ acc_baselineModel1, acc_baselineModel2, acc_normModel1, acc_normModel2, acc_ldaModel1, acc_ldaModel2, acc_ldaPlain ]
fig, ax = plt.subplots( figsize = (12,6) )
modelLabels = ( "Baseline: Small", "Baseline: Large", "Norm: Small", "Norm: Large", "LDA: Small", "LDA: Large", "LDA: Plain" )
ax.barh( np.arange(len(data)), data )
ax.set_yticks( np.arange(len(data)) )
ax.set_yticklabels( modelLabels )
ax.set_xticks( np.arange(0.0,1.05,0.1) )
ax.invert_yaxis()
ax.set_xlabel( "Accuracy" )
ax.set_title( "Accuracy of Models" )
ax.grid()
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Dense, BatchNormalization, Input, Concatenate
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Reshape, Lambda
def make_cnn_model():
#     model = Sequential( [
#         Lambda( lambda x: (x-dataStandardizer.mean_)/(dataStandardizer.scale_), 
#                            input_shape = (784,), output_shape = (784,) ),
#         Reshape( (28,28,1) ),
#         Convolution2D( 32, (5,5), activation = 'relu' ),
#         BatchNormalization( axis = 1 ),
#         Convolution2D( 32, (5,5), activation = 'relu' ),
#         MaxPooling2D(),
#         BatchNormalization( axis = 1 ),
#         Flatten(),
#         Dense( 128, activation = 'tanh' ),
#         Dense( 10, activation = 'softmax' )
#     ])
    
    inputImg = Input( shape = (784,) )
    inputLda = Input( shape = (9,) )
    
    imgBranch = Lambda( lambda x:(x-dataStandardizer.mean_)/(dataStandardizer.scale_), input_shape=(784,), output_shape=(784,) )(inputImg)
    imgBranch = Reshape( (28,28,1) )(imgBranch)
    imgBranch = Convolution2D( 32, (5,5), activation = 'relu' )(imgBranch)
    imgBranch = BatchNormalization( axis = 1 )(imgBranch)
    imgBranch = Convolution2D( 32, (5,5), activation = 'relu' )(imgBranch)
    imgBranch = MaxPooling2D()(imgBranch)
    imgBranch = BatchNormalization( axis = 1 )(imgBranch)
    imgBranch = Convolution2D( 64, (3,3), activation = 'relu' )(imgBranch)
    imgBranch = BatchNormalization( axis = 1 )(imgBranch)
    imgBranch = Convolution2D( 64, (3,3), activation = 'relu' )(imgBranch)
    imgBranch = MaxPooling2D()(imgBranch)
    imgBranch = BatchNormalization( axis = 1 )(imgBranch)
    
    imgBranch = Flatten()(imgBranch)
    
    merged10 = Concatenate()( [imgBranch,inputLda] )
    merged11 = Dense( 512, activation = 'softmax' )(merged10)
    merged12 = Dense( 512, activation = 'relu' )(merged10)
    merged13 = Dense( 512, activation = 'tanh' )(merged10)
    
    merged20 = Concatenate()( [merged11,merged12,merged13,inputLda] )
    merged20 = BatchNormalization()(merged20)
    merged21 = Dense( 128, activation = 'softmax' )(merged20)
    merged22 = Dense( 128, activation = 'relu' )(merged20)
    merged23 = Dense( 128, activation = 'tanh' )(merged20)
    
    merged30 = Concatenate()( [merged21,merged22,merged23,inputLda] )
    merged30 = BatchNormalization()(merged30)
    merged31 = Dense( 128 )(merged30)
    
    output = Dense( 10, activation = 'softmax' )(merged30)
    
    model = Model( inputs = [inputImg,inputLda], outputs = output )
    
    model.compile( Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'] )
    return model
# dataStandardizer.mean_
cnnModel = make_cnn_model()
cnnModel.summary()
# train_model( cnnModel, dataCvTrainFeatRaw, dataCvTrainOnehot, dataCvTestFeatRaw, dataCvTestOnehot, epochs = 5, verbose = 1 )
train_model( cnnModel, [ dataCvTrainFeatRaw, dataCvTrainFeatLDA], dataCvTrainOnehot, 
                       [ dataCvTestFeatRaw, dataCvTestFeatLDA], dataCvTestOnehot, epochs = 10, verbose = 1 )
# from sklearn.metrics import accuracy_score
# acc_cnn            = accuracy_score( dataCvTestLabels, np.argmax(cnnModel.predict([dataCvTestFeatRaw,dataCvTestFeatLDA])) )
from sklearn.metrics import accuracy_score

acc_baselineModel1 = accuracy_score( dataCvTestLabels, np.argmax(baselineModel1.predict(dataCvTestFeatRaw),1) );
acc_baselineModel2 = accuracy_score( dataCvTestLabels, np.argmax(baselineModel2.predict(dataCvTestFeatRaw),1) );

acc_normModel1     = accuracy_score( dataCvTestLabels, np.argmax(normModel1.predict(dataCvTestFeatStd),1) );
acc_normModel2     = accuracy_score( dataCvTestLabels, np.argmax(normModel2.predict(dataCvTestFeatStd),1) );

acc_ldaModel1      = accuracy_score( dataCvTestLabels, np.argmax(ldaModel1.predict(dataCvTestFeatLDA),1) );
acc_ldaModel2      = accuracy_score( dataCvTestLabels, np.argmax(ldaModel2.predict(dataCvTestFeatLDA),1) );

acc_ldaPlain       = accuracy_score( dataCvTestLabels, lda.predict(dataCvTestFeatStd) )

# acc_cnn            = accuracy_score( dataCvTestLabels, np.argmax(cnnModel.predict([dataCvTestFeatRaw,dataCvTestFeatLDA) )


print( "Results" )
print( "Baseline Models: " )
print( "    Small Network    = ", acc_baselineModel1 )
print( "    Large Network    = ", acc_baselineModel2 )
print( "Models with data normalization: " )
print( "    Small Network    = ", acc_normModel1 )
print( "    Large Network    = ", acc_normModel2 )
print( "LDA based Models: " )
print( "    Small Network    = ", acc_ldaModel1 )
print( "    Large Network    = ", acc_ldaModel2 )
print( "Plain LDA            = ", acc_ldaPlain )
# print( "CNN Model            = ", acc_cnn )                                                                                   
                                                                                   


data = [ acc_baselineModel1, acc_baselineModel2, acc_normModel1, acc_normModel2, acc_ldaModel1, acc_ldaModel2, acc_ldaPlain ]
fig, ax = plt.subplots( figsize = (12,6) )
modelLabels = ( "Baseline: Small", "Baseline: Large", "Norm: Small", "Norm: Large", "LDA: Small", "LDA: Large", "LDA: Plain" )
ax.barh( np.arange(len(data)), data )
ax.set_yticks( np.arange(len(data)) )
ax.set_yticklabels( modelLabels )
ax.set_xticks( np.arange(0.0,1.05,0.1) )
ax.invert_yaxis()
ax.set_xlabel( "Accuracy" )
ax.set_title( "Accuracy of Models" )
ax.grid()
lda = LDA()
lda.fit( dataTrainFeatRaw, dataTrainLabels )
dataTrainFeatLDA = lda.transform( dataTrainFeatRaw )
dataTestFeatLDA  = lda.transform( dataTestFeatRaw )

cnnModel = make_cnn_model()
# cnnModel.fit( [dataTrainFeatRaw,dataTrainFeatLDA], dataTrainOnehot , epochs = 3, verbose = 1, batch_size = 128 )
train_model( cnnModel, [ dataTrainFeatRaw, dataTrainFeatLDA], dataTrainOnehot, 
                       None, None, epochs = 50, verbose = 1 )
classProb = cnnModel.predict( [dataTestFeatRaw,dataTestFeatLDA], verbose = 0 )
predictions = classProb.argmax( axis = -1 )
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("results.csv", index=False, header=True)
