import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
# Do this to support SHAP

!pip install 'tensorflow==1.14.0'
variables_name = pd.read_csv("../input/variables_name.csv")

features = variables_name.values[:,1]
features
import json

with open("../input/X_train_HPCC_1_20.json") as of:

    X_train = np.array(json.load(of))

with open("../input/y_train_HPCC_1_20.json") as of:

    y_train = np.array(json.load(of))

with open("../input/X_test_HPCC_1_20.json") as of:

    X_test = np.array(json.load(of))

with open("../input/y_test_HPCC_1_20.json") as of:

    y_test = np.array(json.load(of))    
from keras import regularizers

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Activation

from keras.layers import Dropout

from keras.layers import Flatten

from keras.optimizers import Adam





def createModel(l1Nodes, l2Nodes, d1Nodes, d2Nodes, inputShape):

    # input layer

    lstm1 = LSTM(l1Nodes, input_shape=inputShape, return_sequences=True)

    lstm2 = LSTM(l2Nodes, return_sequences=True)

    flatten = Flatten()

    dense1 = Dense(d1Nodes)

    dense2 = Dense(d2Nodes)



    # output layer

    outL = Dense(1, activation='relu')

    # combine the layers

    layers = [lstm1, lstm2, flatten,  dense1, dense2, outL]

    # create the model

    model = Sequential(layers)

    opt = Adam(learning_rate=0.005)

    model.compile(optimizer=opt, loss='mse')

    return model
# create model

model = createModel(8, 8, 8, 4, (X_train.shape[1], X_train.shape[2]))

model.fit(X_train, y_train, batch_size=8, epochs=30)
import shap
# Use the training data for deep explainer => can use fewer instances

explainer = shap.DeepExplainer(model, X_train)

# explain the the testing instances (can use fewer instanaces)

# explaining each prediction requires 2 * background dataset size runs

shap_values = explainer.shap_values(X_test)

# init the JS visualization code

shap.initjs()

shap.force_plot(explainer.expected_value[0], shap_values[0][0], features)
# X_train_outlier

with open("../input/X_train_outlier.json") as of:

    X_train_outlier = np.array(json.load(of))

with open("../input/y_train_outlier.json") as of:

    y_train_outlier = np.array(json.load(of))



    # X_train_normal

with open("../input/X_train_not_outlier.json") as of:

    X_train_not_outlier = np.array(json.load(of))

with open("../input/y_train_not_outlier.json") as of:

    y_train_not_outlier = np.array(json.load(of))
# explain the the testing instances (can use fewer instanaces)

# explaining each prediction requires 2 * background dataset size runs

shap_values = explainer.shap_values(X_train_outlier)

# init the JS visualization code

shap.initjs()

shap.force_plot(explainer.expected_value[0], shap_values[0][0], features)
# explain the the testing instances (can use fewer instanaces)

# explaining each prediction requires 2 * background dataset size runs

shap_values = explainer.shap_values(X_train_not_outlier)

# init the JS visualization code

shap.initjs()

shap.force_plot(explainer.expected_value[0], shap_values[0][0], features)
# explain the the testing instances (can use fewer instanaces)

# explaining each prediction requires 2 * background dataset size runs

shap_values = explainer.shap_values(X_train_outlier[:1])

# init the JS visualization code

shap.initjs()

shap.force_plot(explainer.expected_value[0], shap_values[0][0], features)
# explain the the testing instances (can use fewer instanaces)

# explaining each prediction requires 2 * background dataset size runs

shap_values = explainer.shap_values(X_train_not_outlier[:1])

# init the JS visualization code

shap.initjs()

shap.force_plot(explainer.expected_value[0], shap_values[0][0], features)
y_train_not_outlier[0]
y_train_outlier[0]