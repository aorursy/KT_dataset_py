import numpy as np

import pandas as pd



from keras.models import Sequential, load_model

from keras.layers import Dense, BatchNormalization, Dropout

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras.optimizers import RMSprop

from keras import backend as K

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score



import skopt

from skopt import gp_minimize, forest_minimize

from skopt.space import Real, Categorical, Integer

from skopt.utils import use_named_args



import os
print(os.listdir("../input"))
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
x_train = train_data.drop(['id', 'target'], axis = 1)

y_train = train_data['target']
del train_data
test_data = test_data.drop(['id'], axis = 1)
y_train.value_counts()
x_train.shape
x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size = 0.2, stratify = y_train, random_state = 42)
dim_num_dense_layers = Integer(low = 1, high = 5, name='num_dense_layers')

dim_num_dense_nodes = Integer(low = 5, high = 512, name='num_dense_nodes')

dim_activation = Categorical(categories = ['relu', 'tanh'], name = 'activation')

dim_optimizer = Categorical(categories = ['rmsprop', 'adam', 'nadam'], name = 'optimizer')
dimensions = [dim_num_dense_layers,

              dim_num_dense_nodes,

              dim_activation,

              dim_optimizer]
default_parameters = [3, 200, 'relu', 'rmsprop']
def create_model(num_dense_layers, num_dense_nodes, activation, optimizer):

    model = Sequential()

    

    model.add(Dense(300, input_shape = (300,)))

    

    

    for i in range(num_dense_layers):

        name = 'layer_dense_{0}'.format(i + 1)

        

        model.add(Dense(num_dense_nodes, activation = activation, name = name))

    

    model.add(Dropout(0.5))

    

    model.add(Dense(1, activation = 'sigmoid'))

    

    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    

    return model
path_best_model = 'best_model.keras'



best_accuracy = 0.0



best_roc_auc = 0.0
@use_named_args(dimensions = dimensions)

def fitness(num_dense_layers, num_dense_nodes, activation, optimizer):

    

    reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_acc', patience = 3, verbose = 0, factor = 0.75, min_lr = 0.00001)

    early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 1e-10, patience = 10, verbose = 1, restore_best_weights = True)



    callbacks = [reduce_learning_rate, early_stopping]

    

    batch_size = 8



    train_step_size = x_train.shape[0] // batch_size



    print('Num dense layers: ', num_dense_layers)

    print('Num dense nodes: ', num_dense_nodes)

    print('Activation: ', activation)

    print('Optimizer: ', optimizer)

    print()

    

    model = create_model(num_dense_layers = num_dense_layers,

                         num_dense_nodes = num_dense_nodes,

                         activation = activation,

                         optimizer = optimizer)

    

    history = model.fit(x_tr, 

                        y_tr,

                        epochs = 50,

                        validation_data = (x_val, y_val),

                        verbose = 0,

                        callbacks = callbacks)

    

    roc_auc = roc_auc_score(y_val, model.predict(x_val))

    

    print()

    print("ROC AUC: {0:.2%}".format(roc_auc))

    print()

    

    global best_roc_auc

    

    if roc_auc > best_roc_auc:

        model.save(path_best_model)

        

        best_roc_auc = roc_auc

    

    #accuracy = history.history['val_acc'][-1]

    

    #print()

    #print("Accuracy: {0:.2%}".format(accuracy))

    #print()

    

    #global best_accuracy

    

    #if accuracy > best_accuracy:

    #    model.save(path_best_model)

    #    

    #    best_accuracy = accuracy

    

    del model

    

    K.clear_session()

    

    return -roc_auc

        
search_result = gp_minimize(func = fitness, dimensions = dimensions, acq_func = 'EI', n_calls = 100, x0 = default_parameters)
search_result.x
search_result.fun
predict_model = load_model(path_best_model)



predictions = predict_model.predict(test_data)
ps = []

for i, value in enumerate(predictions):

    ps.append(value[0])



#predictions = np.argmax(predictions, axis = 1)



predictions = pd.Series(ps, name = "target")



submission = pd.concat([pd.Series(range(250, 20000), name = "id"), predictions], axis = 1)



submission.to_csv("dont-overfit-submission.csv", index = False)