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
trainset0 = pd.read_csv('../input/Dataset 002_train data without normalization.csv')
testset0 = pd.read_csv('../input/Dataset 002_test data without normalization.csv')
trainset1 = pd.read_csv('../input/Dataset 002_train data after normalization.csv')
testset1 = pd.read_csv('../input/Dataset 002_test data after normalization.csv')
import matplotlib.pyplot as plt
import math
# Display methods
import matplotlib.pyplot as plt
import math

def plotUnitLines(y_pred, startUnit, endUnit, ncols=4, title='', figsize=[15, 15]):
    nunits = endUnit - startUnit + 1
    nrows = math.ceil(nunits/ncols)
    fig, axes=plt.subplots(nrows=nrows, ncols=ncols, clear=True, figsize=figsize)
    for unit in range(startUnit, endUnit+1):
        unitBooleanIndexes = testset0['unit']==unit;
        unit_pred = list(map(lambda x: x[0], y_pred[unitBooleanIndexes]))
        length = len(unit_pred)
        #Get the rul for each unit based on its unit boolean index
        y_rul = y_test0[unitBooleanIndexes].values
        mse = mean_squared_error(y_rul, unit_pred)
        x = range(len(unit_pred))
        rowIndex = math.floor((unit-startUnit)/ncols)
        colIndex = (unit-startUnit)%ncols;
        ax = axes[rowIndex, colIndex]
        ax.plot(x, unit_pred, label= 'predicted')
        ax.plot(x, y_rul, label='actual')
        ax.set_title('Unit'+str(unit) + ', MSE: '+str(mse))
        ax.legend()
    fig.tight_layout()
    plt.show()
def plotLossHistory(model):
    plt.plot(model.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
def plotTrainHistory(model):
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Training history')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper right')
    plt.show()    
#RBF adopted from: https://github.com/PetraVidnerova/rbf_keras
import random 
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Orthogonal, Constant
import numpy as np

class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows 
          are taken as centers)
    """
    def __init__(self, X):
        self.X = X 

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx,:]

        
class RBFLayer(Layer):
    """ Layer of Gaussian RBF units. 
    # Example
 
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X), 
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas 
    """
    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas 
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
            #self.initializer = Orthogonal()
        else:
            self.initializer = initializer 
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.centers = self.add_weight(name='centers', 
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(value=self.init_betas),
                                     #initializer='ones',
                                     trainable=True)
            
        super(RBFLayer, self).build(input_shape)  

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp( -self.betas * K.sum(H**2, axis=1))
        
        #C = self.centers[np.newaxis, :, :]
        #X = x[:, np.newaxis, :]

        #diffnorm = K.sum((C-X)**2, axis=-1)
        #ret = K.exp( - self.betas * diffnorm)
        #return ret 

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
def executeModel(inputTrainset, inputTestset, columnsToDrop, numOfNodes=100, epochs=100, batch_size = None, verbose=True, isRBF=False, validationSplit=0):
    print('==============================================================================')
    print ('Delete columns: ')
    print (columnsToDrop)
    print ('Num of ndoes: ' + str(numOfNodes))
    print ('Epochs: ' + str(epochs))
    print('==============================================================================')
    trainset = inputTrainset
    testset = inputTestset
    #Remove columns
    trainset = trainset.drop(columnsToDrop, axis = 1)
    testset = testset.drop(columnsToDrop, axis=1)
    #y_train, X_train
    y_train = trainset['RUL'].values
    X_train = trainset.drop('RUL', axis=1)
    #Converting to array
    X_train = X_train.values

    #y_test, X_test
    y_test0 = testset['RUL']
    y_test = y_test0.values
    X_test = testset.drop('RUL', axis=1)
    #Converting to array
    X_test = X_test.values
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Building the model
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import RMSprop

    model = Sequential()
    layer1 = Dense(units=numOfNodes, activation='relu', input_dim = len(X_train[0]))
    layer2 = Dense(units=1, activation='relu')
    optimizer = 'adam'
    if(isRBF):
        layer1 = RBFLayer(10,
                        initializer=InitCentersRandom(X_train), 
                        betas=2.0,
                        input_shape=(len(X_train[0]),))
        layer2 = Dense(units=1)
        optimizer = RMSprop()
    model.add(layer1)
    model.add(layer2)
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    #Train the model
    history=None
    if(validationSplit==0):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    else:
        history = model.fit(X_train, y_train, validation_split=validationSplit, epochs=epochs, batch_size=batch_size, verbose=verbose)
    #Predict
    y_pred_model = model.predict(X_test)
    print('THE MODEL TRAIN SCORE IS: ' +str(history.history['loss'][-1]) + ', TEST SCORE IS: ' +str(calculateScore(y_pred_model, y_test)))
    return (y_pred_model, history)
from sklearn.metrics import mean_squared_error
def calculateScore(y_pred, y_test):
    y_pred  = list(map(lambda x: x[0], y_pred))
    return mean_squared_error(y_pred, y_test)
deleteCols = ['Unnamed: 0', 'unit']
clsCol = ['cls']
modeCols = ['mode1', 'mode2', 'mode3', 'mode4', 'mode5', 'mode6']
featureCols = ['feature2', 'feature3', 'feature4', 'feature5', 'feature9', 'feature10', 'feature11', 'feature13', 'feature14', 'feature15', 'feature16', 'feature17', 'feature19', 'feature20', 'feature21']
featureCols1 = ['feature1', 'feature2', 'feature3', 'feature4', 'feature7', 'feature10', 'feature11', 'feature12', 'feature14', 'feature15', 'feature16', 'feature17', 'feature18', 'feature19', 'feature20', 'feature21']
cycleCol = ['cycle']
settingCols = ['setting1', 'setting2', 'setting3']
#The test RUL => for trainset or trainset1 are the same so just do for one
y_test0 = testset0['RUL']
y_test = y_test0.values

# print('MODEL: 1 HIDDEN LAYER, 100 NODES, 100 EPOCHS')
# y_pred_mod1, history1 = executeModel(trainset0, testset0, deleteCols, 100, 100, None, False, False)
# y_pred_mod2, history2 = executeModel(trainset0, testset0, deleteCols+clsCol, 100, 100, None, False, False)
# y_pred_mod3, history3 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 100, 100, None, False, False)
# y_pred_mod4, history4 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+cycleCol, 100, 100, None, False, False)
# y_pred_mod5, history5 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+cycleCol+settingCols, 100, 100,None, False, False)

# print('MODEL: 1 HIDDEN LAYER, 100 NODES, 200 EPOCHS')
# y_pred_mod6, history6 = executeModel(trainset0, testset0, deleteCols, 100, 200, None, False, False)
# y_pred_mod7, history7 = executeModel(trainset0, testset0, deleteCols+clsCol, 100, 200, None, False, False)
# y_pred_mod8, history8 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 100, 200, None, False, False)
# y_pred_mod9, history9 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+cycleCol, 100, 200, None, False, False)
# y_pred_mod10, history10= executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+cycleCol+settingCols, 100, 200,None, False, False)

# print('MODEL: 1 HIDDEN LAYER, 100 NODES, 300 EPOCHS')
# y_pred_mod11, history11 = executeModel(trainset0, testset0, deleteCols, 100, 300, None, False, False)
# y_pred_mod12, history12 = executeModel(trainset0, testset0, deleteCols+clsCol, 100, 300, None, False, False)
# y_pred_mod13, history13 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 100, 300, None, False, False)
# y_pred_mod14, history14 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+cycleCol, 100, 300, None, False, False)
# y_pred_mod15, history15 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+cycleCol+settingCols, 100, 300, None, False, False)

# print('MODEL: 1 HIDDEN LAYER, 100 NODES, 100 EPOCHS, NON TRAIN DATA NORMALIZED BY CLASSES')
# y_pred_mod16, history16 = executeModel(trainset1, testset1, deleteCols, 100, 100, None, False, False)
# y_pred_mod17, history17 = executeModel(trainset1, testset1, deleteCols+clsCol, 100, 100, None, False, False)
# y_pred_mod18, history18 = executeModel(trainset1, testset1, deleteCols+clsCol+modeCols, 100, 100, None, False, False)
# y_pred_mod19, history19 = executeModel(trainset1, testset1, deleteCols+clsCol+modeCols+cycleCol, 100, 100, None, False, False)
# y_pred_mod20, history20 = executeModel(trainset1, testset1, deleteCols+clsCol+modeCols+cycleCol+settingCols, 100, 100,None, False, False)

# print('MODEL: 1 HIDDEN LAYER, 100 NODES, 200 EPOCHS, NON TRAIN DATA NORMALIZED BY CLASSES')
# y_pred_mod21, history21 = executeModel(trainset1, testset1, deleteCols, 100, 200, None, False, False)
# y_pred_mod22, history22 = executeModel(trainset1, testset1, deleteCols+clsCol, 100, 200, None, False, False)
# y_pred_mod23, history23 = executeModel(trainset1, testset1, deleteCols+clsCol+modeCols, 100, 200, None, False, False)
# y_pred_mod24, history24 = executeModel(trainset1, testset1, deleteCols+clsCol+modeCols+cycleCol, 100, 200, None, False, False)
# y_pred_mod25, history25 = executeModel(trainset1, testset1, deleteCols+clsCol+modeCols+cycleCol+settingCols, 100, 200, None, False, False)

# print('MODEL: 1 HIDDEN LAYER, 100 NODES, 300 EPOCHS, NON TRAIN DATA NORMALIZED BY CLASSES')
# y_pred_mod26, history26 = executeModel(trainset1, testset1, deleteCols, 100, 300, None, False,False)
# y_pred_mod27, history27 = executeModel(trainset1, testset1, deleteCols+clsCol, 100, 300, None, False, False)
# y_pred_mod28, history28 = executeModel(trainset1, testset1, deleteCols+clsCol+modeCols, 100, 300, None, False, False)
# y_pred_mod29, history29 = executeModel(trainset1, testset1, deleteCols+clsCol+modeCols+cycleCol, 100, 300, None, False, False)
# y_pred_mod30, history30 = executeModel(trainset1, testset1, deleteCols+clsCol+modeCols+cycleCol+settingCols, 100, 300, None, False, False)
# plotLossHistory(history1)
# plotLossHistory(history2)
# plotLossHistory(history3)
# plotLossHistory(history4)
# plotLossHistory(history5)
# plotLossHistory(history6)
# plotLossHistory(history7)
# plotLossHistory(history8)
# plotLossHistory(history9)
# plotLossHistory(history10)
# plotLossHistory(history11)
# plotLossHistory(history12)
# plotLossHistory(history13)
# plotLossHistory(history14)
# plotLossHistory(history15)
# plotLossHistory(history16)
# plotLossHistory(history17)
# plotLossHistory(history18)
# plotLossHistory(history19)
# plotLossHistory(history20)
# plotLossHistory(history21)
# plotLossHistory(history22)
# plotLossHistory(history23)
# plotLossHistory(history24)
# plotLossHistory(history25)
# plotLossHistory(history26)
# plotLossHistory(history27)
# plotLossHistory(history28)
# plotLossHistory(history29)
# plotLossHistory(history30)
# plotUnitLines(y_pred_mod1, 1, 20, ncols=4, title='Model 1', figsize=[15, 15])
# plotUnitLines(y_pred_mod2, 1, 20, ncols=4, title='Model 2', figsize=[15, 15])
# plotUnitLines(y_pred_mod3, 1, 20, ncols=4, title='Model 3', figsize=[15, 15])
# plotUnitLines(y_pred_mod4, 1, 20, ncols=4, title='Model 4', figsize=[15, 15])
# plotUnitLines(y_pred_mod5, 1, 20, ncols=4, title='Model 5', figsize=[15, 15])
# plotUnitLines(y_pred_mod6, 1, 20, ncols=4, title='Model 6', figsize=[15, 15])
# plotUnitLines(y_pred_mod7, 1, 20, ncols=4, title='Model 7', figsize=[15, 15])
# plotUnitLines(y_pred_mod8, 1, 20, ncols=4, title='Model 8', figsize=[15, 15])
# plotUnitLines(y_pred_mod9, 1, 20, ncols=4, title='Model 9', figsize=[15, 15])
# plotUnitLines(y_pred_mod10, 1, 20, ncols=4, title='Model 10', figsize=[15, 15])
# plotUnitLines(y_pred_mod11, 1, 20, ncols=4, title='Model 11', figsize=[15, 15])
# plotUnitLines(y_pred_mod12, 1, 20, ncols=4, title='Model 12', figsize=[15, 15])
# plotUnitLines(y_pred_mod13, 1, 20, ncols=4, title='Model 13', figsize=[15, 15])
# plotUnitLines(y_pred_mod14, 1, 20, ncols=4, title='Model 14', figsize=[15, 15])
# plotUnitLines(y_pred_mod15, 1, 20, ncols=4, title='Model 15', figsize=[15, 15])
# plotUnitLines(y_pred_mod16, 1, 20, ncols=4, title='Model 16', figsize=[15, 15])
# plotUnitLines(y_pred_mod17, 1, 20, ncols=4, title='Model 17', figsize=[15, 15])
# plotUnitLines(y_pred_mod18, 1, 20, ncols=4, title='Model 18', figsize=[15, 15])
# plotUnitLines(y_pred_mod19, 1, 20, ncols=4, title='Model 19', figsize=[15, 15])
# plotUnitLines(y_pred_mod20, 1, 20, ncols=4, title='Model 20', figsize=[15, 15])
# plotUnitLines(y_pred_mod21, 1, 20, ncols=4, title='Model 21', figsize=[15, 15])
# plotUnitLines(y_pred_mod22, 1, 20, ncols=4, title='Model 22', figsize=[15, 15])
# plotUnitLines(y_pred_mod23, 1, 20, ncols=4, title='Model 23', figsize=[15, 15])
# plotUnitLines(y_pred_mod24, 1, 20, ncols=4, title='Model 24', figsize=[15, 15])
# plotUnitLines(y_pred_mod25, 1, 20, ncols=4, title='Model 25', figsize=[15, 15])
# plotUnitLines(y_pred_mod26, 1, 20, ncols=4, title='Model 26', figsize=[15, 15])
# plotUnitLines(y_pred_mod27, 1, 20, ncols=4, title='Model 27', figsize=[15, 15])
# plotUnitLines(y_pred_mod28, 1, 20, ncols=4, title='Model 28', figsize=[15, 15])
# plotUnitLines(y_pred_mod29, 1, 20, ncols=4, title='Model 29', figsize=[15, 15])
# plotUnitLines(y_pred_mod30, 1, 20, ncols=4, title='Model 30', figsize=[15, 15])
# print('MODEL: 1 HIDDEN LAYER, 100 NODES, 100 EPOCHS, WITH FEATURE SELECTION 1')
# y_pred_mod31, history31 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+featureCols, 100, 100, None, False, False)
# print('MODEL: 1 HIDDEN LAYER, 100 NODES, 200 EPOCHS, WITH FEATURE SELECTION 1')
# y_pred_mod32, history32 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+featureCols, 100, 200, None, False, False)
# print('MODEL: 1 HIDDEN LAYER, 100 NODES, 100 EPOCHS, WITH FEATURE SELECTION 2')
# y_pred_mod33, history33 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+featureCols1, 100, 100, None, False, False)
# print('MODEL: 1 HIDDEN LAYER, 100 NODES, 200 EPOCHS, WITH FEATURE SELECTION 2')
# y_pred_mod34, history34 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols+featureCols1, 100, 200, None, False, False)
# plotLossHistory(history31)
# plotLossHistory(history32)
# plotLossHistory(history33)
# plotLossHistory(history34)
# plotUnitLines(y_pred_mod31, 1, 20, ncols=4, title='Model 31', figsize=[15, 15])
# plotUnitLines(y_pred_mod32, 1, 20, ncols=4, title='Model 32', figsize=[15, 15])
# plotUnitLines(y_pred_mod33, 1, 20, ncols=4, title='Model 33', figsize=[15, 15])
# plotUnitLines(y_pred_mod34, 1, 20, ncols=4, title='Model 34', figsize=[15, 15])
# y_pred35, history35 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 15, 100, None, False, True)
# y_pred36, history36 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 15, 200, None, False, True)
# y_pred37, history37 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 15, 300, None, False, True)
# y_pred44, history44 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 15, 40, None, False, True)
# y_pred47, history47 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 15, 50, None, False, True)

# y_pred38, history38 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 100, 100, None, False, False)
# y_pred39, history39 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 100, 200, None, False, False)
# y_pred40, history40 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 100, 300, None, False, False)
# y_pred45, history45 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 100, 40, None, False, False)
# y_pred48, history48 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 15, 50, None, False, True)

# y_pred41, history41 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 75, 100, None, False, False)
# y_pred42, history42 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 75, 200, None, False, False)
# y_pred43, history43 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 75, 300, None, False, False)
# y_pred46, history46 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 75, 40, None, False, False)
# y_pred49, history49 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 15, 50, None, False, True)
#ensemble
# y_pred353841 = (y_pred35 + y_pred38 + y_pred41)/3
# y_pred363942 = (y_pred36 + y_pred39 + y_pred42)/3
# y_pred374043 = (y_pred37 + y_pred40 + y_pred43)/3
# y_pred444546 = (y_pred44 + y_pred45 + y_pred46)/3
# y_pred474849 = (y_pred47 + y_pred48 + y_pred49)/3
#calculate score
# print("Score 353841: " + str(calculateScore(y_pred353841, y_test)))
# print("Score 363942: " + str(calculateScore(y_pred363942, y_test)))
# print("Score 374043: " + str(calculateScore(y_pred374043, y_test)))
# print("Score 444546: " + str(calculateScore(y_pred444546, y_test)))
# print("Score 474849: " + str(calculateScore(y_pred474849, y_test)))
# plotLossHistory(history35)
# plotLossHistory(history36)
# plotLossHistory(history37)
# plotLossHistory(history44)
# plotLossHistory(history47)
# plotLossHistory(history38)
# plotLossHistory(history39)
# plotLossHistory(history40)
# plotLossHistory(history45)
# plotLossHistory(history48)
# plotLossHistory(history41)
# plotLossHistory(history42)
# plotLossHistory(history43)
# plotLossHistory(history46)
# plotLossHistory(history49)
# plotUnitLines(y_pred35, 1, 20, ncols=4, title='Model 35', figsize=[15, 15])
# plotUnitLines(y_pred36, 1, 20, ncols=4, title='Model 36', figsize=[15, 15])
# plotUnitLines(y_pred37, 1, 20, ncols=4, title='Model 37', figsize=[15, 15])
# plotUnitLines(y_pred38, 1, 20, ncols=4, title='Model 38', figsize=[15, 15])
# plotUnitLines(y_pred39, 1, 20, ncols=4, title='Model 39', figsize=[15, 15])
# plotUnitLines(y_pred40, 1, 20, ncols=4, title='Model 40', figsize=[15, 15])
# plotUnitLines(y_pred41, 1, 20, ncols=4, title='Model 41', figsize=[15, 15])
# plotUnitLines(y_pred42, 1, 20, ncols=4, title='Model 42', figsize=[15, 15])
# plotUnitLines(y_pred43, 1, 20, ncols=4, title='Model 43', figsize=[15, 15])
# plotUnitLines(y_pred44, 1, 20, ncols=4, title='Model 44', figsize=[15, 15])
# plotUnitLines(y_pred45, 1, 20, ncols=4, title='Model 45', figsize=[15, 15])
# plotUnitLines(y_pred46, 1, 20, ncols=4, title='Model 46', figsize=[15, 15])
# plotUnitLines(y_pred47, 1, 20, ncols=4, title='Model 47', figsize=[15, 15])
# plotUnitLines(y_pred48, 1, 20, ncols=4, title='Model 48', figsize=[15, 15])
# plotUnitLines(y_pred49, 1, 20, ncols=4, title='Model 49', figsize=[15, 15])

# plotUnitLines(y_pred353841, 1, 20, ncols=4, title='Model 353841', figsize=[15, 15])
# plotUnitLines(y_pred363942, 1, 20, ncols=4, title='Model 363942', figsize=[15, 15])
# plotUnitLines(y_pred374043, 1, 20, ncols=4, title='Model 374043', figsize=[15, 15])
# plotUnitLines(y_pred444546, 1, 20, ncols=4, title='Model 444', figsize=[15, 15])
# plotUnitLines(y_pred474849, 1, 20, ncols=4, title='Model 474849', figsize=[15, 15])
# y_pred50, history50 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 15, 300, None, False, True, 0.33)
# y_pred51, history51 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 100, 300, None, False, False, 0.33)
#Try RBF with 100 nodes
y_pred52, history52 = executeModel(trainset0, testset0, deleteCols+clsCol+modeCols, 100, 300, None, False, True, 0.33)
# plotTrainHistory(history50)
# plotTrainHistory(history51)
plotTrainHistory(history52)
# plotUnitLines(y_pred50, 1, 20, ncols=4, title='Model 50', figsize=[15, 15])
# plotUnitLines(y_pred51, 1, 20, ncols=4, title='Model 51', figsize=[15, 15])
plotUnitLines(y_pred52, 1, 20, ncols=4, title='Model 52', figsize=[15, 15])
