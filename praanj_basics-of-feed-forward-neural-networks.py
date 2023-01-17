import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from keras import backend as K

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import f1_score

from tensorflow.keras.utils import plot_model



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%matplotlib inline
dataset = pd.read_csv('../input/uci-credit-approval-data-set/UCI_crx.csv')

dataset.shape
dataset.head()
dataset.dtypes
for col in ['A1', 'A2','A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13', 'A14', 'A16']:

    dataset[col] = pd.Categorical(dataset[col])

    dataset[col] = dataset[col].cat.codes
dataset.dtypes
dataset.head()
# create scaler

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

dataset = pd.DataFrame(scaler.fit_transform(dataset))



dataset.describe()
X=dataset.iloc[:,0:15].values   #0:15

Y=dataset.iloc[:,15:16].values
np.random.seed(42) # Makes the random numbers predictable for easy comparison of models
## Set these parameter before calling create_model function

depthOfNetwork = 3

neuronCountInEachLayer = [16, 9, 1]                                 # try different depth and width

activationFuncEachLayer = ['sigmoid', 'relu', 'sigmoid']            # try values relu, sigmoid, talh

lossFunction = 'binary_crossentropy'                                # try values binary_crossentropy, mean_squared_error

regularizerFunc = tf.keras.regularizers.l2(0)                       # try l1 and l2 with different lambda





def create_model(verbose=False):

  model = tf.keras.models.Sequential()

  

  if verbose:

        print('Network configuration ',neuronCountInEachLayer)

  

  model.add(tf.keras.layers.Dense(neuronCountInEachLayer[0], input_dim=15, activation = activationFuncEachLayer[0], kernel_regularizer=regularizerFunc)) # First Layer

    

  for x in range(1, depthOfNetwork-1):

      model.add(tf.keras.layers.Dense(neuronCountInEachLayer[x], activation = activationFuncEachLayer[x],kernel_regularizer=regularizerFunc))         # Second layer onwards

 

  model.add(tf.keras.layers.Dense(neuronCountInEachLayer[depthOfNetwork-1], activation = activationFuncEachLayer[depthOfNetwork-1]))  # Output layer

    

  model.compile(loss = lossFunction , optimizer = 'adam' , metrics = ['accuracy'] ) 

        

  return model
depthOfNetwork = 3

neuronCountInEachLayer = [17, 8, 1]                                 # try different depth and width

activationFuncEachLayer = ['sigmoid', 'relu', 'sigmoid']            # try values relu, sigmoid, talh

lossFunction = 'binary_crossentropy'                                # try values binary_crossentropy, mean_squared_error

regularizerFunc = tf.keras.regularizers.l2(0)                       # try l1 and l2 with different lambda



model=create_model()

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
def evaluateTheModel(verbose=False):

    n_split=5

    f1_scores = []



    for train_index,test_index in StratifiedKFold(n_split).split(X, Y):      # StratifiedKFold, KFold

        x_train,x_test=X[train_index],X[test_index]

        y_train,y_test=Y[train_index],Y[test_index]



        model=create_model(verbose)

        model.fit(x_train, y_train,epochs=100, verbose=0)

        evaluationMetrics = model.evaluate(x_test,y_test, verbose=0)

        

        if verbose:

            print('Model evaluation ',evaluationMetrics)   # This returns metric values for the evaluation



        y_pred = np.where(model.predict(x_test) > 0.5, 1, 0)

        f1 = f1_score(y_test, y_pred , average="macro")



        if verbose:

            print('F1 score is ', f1)

        

        f1_scores.append(f1)

    

    return np.mean(f1_scores)
depthOfNetwork = 2

neuronCountInEachLayer = [2, 1]                                 # try different depth and width

activationFuncEachLayer = ['sigmoid', 'sigmoid']            # try values relu, sigmoid, talh

lossFunction = 'binary_crossentropy'                                # try values binary_crossentropy, mean_squared_error

regularizerFunc = tf.keras.regularizers.l2(0)                       # try l1 and l2 with different lambda



evaluateTheModel(True)
depthOfNetwork = 2

neuronCountInEachLayer = [15, 1]                                 # try different depth and width

activationFuncEachLayer = ['sigmoid', 'sigmoid']            # try values relu, sigmoid, talh

lossFunction = 'binary_crossentropy'                                # try values binary_crossentropy, mean_squared_error

regularizerFunc = tf.keras.regularizers.l2(0)                       # try l1 and l2 with different lambda



for i in range (3, 20):

    neuronCountInEachLayer = [i, 1]

    print("'Node count : % 3d, Mean F1 score : % 10.5f" %(i, evaluateTheModel())) 
depthOfNetwork = 2

neuronCountInEachLayer = [15, 1]                                 # try different depth and width

activationFuncEachLayer = ['relu', 'sigmoid']            # try values relu, sigmoid, talh

lossFunction = 'binary_crossentropy'                                # try values binary_crossentropy, mean_squared_error

regularizerFunc = tf.keras.regularizers.l2(0)                       # try l1 and l2 with different lambda



for i in range (3, 20):

    neuronCountInEachLayer = [i, 1]

    print("'Node count : % 3d, Mean F1 score : % 10.5f" %(i, evaluateTheModel())) 
depthOfNetwork = 2

neuronCountInEachLayer = [15, 1]                                 # try different depth and width

activationFuncEachLayer = ['tanh', 'sigmoid']            # try values relu, sigmoid, talh

lossFunction = 'binary_crossentropy'                                # try values binary_crossentropy, mean_squared_error

regularizerFunc = tf.keras.regularizers.l2(0)                       # try l1 and l2 with different lambda



for i in range (3, 20):

    neuronCountInEachLayer = [i, 1]

    print("'Node count : % 3d, Mean F1 score : % 10.5f" %(i, evaluateTheModel())) 
depthOfNetwork = 2

neuronCountInEachLayer = [15, 1]                                 # try different depth and width

activationFuncEachLayer = ['tanh', 'sigmoid']            # try values relu, sigmoid, talh

lossFunction = 'mean_squared_error'                                # try values binary_crossentropy, mean_squared_error

regularizerFunc = tf.keras.regularizers.l2(0)                       # try l1 and l2 with different lambda



for i in range (3, 20):

    neuronCountInEachLayer = [i, 1]

    print("'Node count : % 3d, Mean F1 score : % 10.5f" %(i, evaluateTheModel())) 
depthOfNetwork = 3

neuronCountInEachLayer = [18, 9, 1]                                 # try different depth and width

activationFuncEachLayer = ['tanh', 'tanh', 'sigmoid']            # try values relu, sigmoid, talh

lossFunction = 'binary_crossentropy'                                # try values binary_crossentropy, mean_squared_error

regularizerFunc = tf.keras.regularizers.l2(0)                       # try l1 and l2 with different lambda



for i in range (15, 16):

    for j in range (3, 20):

        neuronCountInEachLayer = [i, j, 1]

        print("'Neurons [% 3d, % 3d], Mean F1 score : % 10.5f" %(i, j, evaluateTheModel())) 
depthOfNetwork = 4

neuronCountInEachLayer = [15, 8, 5, 1]                                 # try different depth and width

activationFuncEachLayer = ['tanh', 'tanh', 'tanh','sigmoid']            # try values relu, sigmoid, talh

lossFunction = 'binary_crossentropy'                                # try values binary_crossentropy, mean_squared_error

regularizerFunc = tf.keras.regularizers.l2(0)                       # try l1 and l2 with different lambda



print("'Neurons [% 3d, % 3d, % 3d], Mean F1 score : % 10.5f" %(3, 4, 3, evaluateTheModel())) 
depthOfNetwork = 2

neuronCountInEachLayer = [15, 1]                                 # try different depth and width

activationFuncEachLayer = ['tanh', 'sigmoid']            # try values relu, sigmoid, talh

lossFunction = 'binary_crossentropy'                                # try values binary_crossentropy, mean_squared_error

regularizerFunc = tf.keras.regularizers.l2(0)                       # try l1 and l2 with different lambda



print("'Mean F1 score : % 10.5f" %(evaluateTheModel())) 
depthOfNetwork = 2

neuronCountInEachLayer = [15,1]                                 # try different depth and width

activationFuncEachLayer = ['tanh', 'sigmoid']            # try values relu, sigmoid, talh

lossFunction = 'binary_crossentropy'                                # try values binary_crossentropy, mean_squared_error

regularizerFunc = tf.keras.regularizers.l1(0)                       # try l1 and l2 with different lambda



for i in range(-5,5):

    regularizerFunc = tf.keras.regularizers.l1(10**i)

    print("'Regularizor : l1 with lambda : % 10.5f , Mean F1 score : % 10.5f" %(10**i, evaluateTheModel()))
depthOfNetwork = 2

neuronCountInEachLayer = [15,1]                                 # try different depth and width

activationFuncEachLayer = ['tanh', 'sigmoid']            # try values relu, sigmoid, talh

lossFunction = 'binary_crossentropy'                                # try values binary_crossentropy, mean_squared_error

regularizerFunc = tf.keras.regularizers.l2(0)                       # try l1 and l2 with different lambda



for i in range(-5,5):

    regularizerFunc = tf.keras.regularizers.l2(10**i)

    print("'Regularizor : l2 with lambda : % 10.5f , Mean F1 score : % 10.5f" %(10**i, evaluateTheModel()))