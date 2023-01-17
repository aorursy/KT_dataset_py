import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
print(os.listdir("../input/stock-data"))
!pip install tensorflow==1.14.0
import json

with open("../input/stock-data/X_train_stock_ts4_fri.json") as of:

    X_train = np.array(json.load(of))

with open("../input/stock-data/y_train_stock_ts4_fri.json") as of:

    y_train = np.array(json.load(of))

with open("../input/stock-data/X_test_stock_ts4_fri.json") as of:

    X_test = np.array(json.load(of))

with open("../input/stock-data/y_test_stock_ts4_fri.json") as of:

    y_test = np.array(json.load(of))    
X_train.shape, y_train.shape
X_test.shape, y_test.shape
# Set dummy names for stock features

features = ['feature_' + str(i) for i in range(1, X_test.shape[-1]+1)]

features
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

#     outL = Dense(1, activation='relu')

    outL = Dense(1)

    # combine the layers

    layers = [lstm1, lstm2, flatten,  dense1, dense2, outL]

    # create the model

    model = Sequential(layers)

    opt = Adam(learning_rate=0.005)

    model.compile(optimizer=opt, loss='mse')

    return model
# create model

model = createModel(12, 8, 8, 4, (X_train.shape[1], X_train.shape[2]))

model.fit(X_train, y_train, batch_size=8, epochs=30)
from sklearn.metrics import mean_squared_error as mse
y_pred = model.predict(X_test)

mse(y_test, y_pred)
# Save the entire model to a HDF5 file.

# The '.h5' extension indicates that the model shuold be saved to HDF5.

model.save('stock_model.h5') 
model.summary()
import shap
import tensorflow as tf

tf.__version__
# Use the training data for deep explainer => can use fewer instances

explainer_ = shap.DeepExplainer(model, X_train)

shap_values_ = explainer_.shap_values(X_test)

# init the JS visualization code

shap.initjs()
# Use the training data for deep explainer => can use fewer instances

explainer_stock = shap.GradientExplainer(model, X_train)

# explain the the testing instances (can use fewer instanaces)

# explaining each prediction requires 2 * background dataset size runs

shap_values_stock = explainer_stock.shap_values(X_test)

# init the JS visualization code

shap.initjs()
################# Plot AVERAGE shap values for ALL observations  #####################

## Consider ABSOLUTE of SHAP values ##

shap_average_abs_value_stock = np.abs(shap_values_stock[0]).mean(axis=0)



x_average_value = pd.DataFrame(data=X_test.mean(axis=0), columns = features)

shap.force_plot(0, shap_average_abs_value_stock, x_average_value)
shap.__version__
shap_values_stock[0][0].shape
(y_pred.sum() -  shap_values_stock[0].sum())/  len(y_pred)
shap_values_2D = shap_values_stock[0].reshape(-1,5)

X_test_2D = X_test.reshape(-1,5)





shap_values_2D.shape, X_test_2D.shape
x_test_2d = pd.DataFrame(data=X_test_2D, columns = features)
x_test_2d.corr()
shap.summary_plot(shap_values_2D, x_test_2d)
shap.summary_plot(shap_values_2D, x_test_2d, plot_type="bar")
len_test_set = X_test_2D.shape[0]

len_test_set
## SHAP for each time step

NUM_STEPS = 4

NUM_FEATURES = 5





# step = 0

for step in range(NUM_STEPS):

    indice = [i for i in list(range(len_test_set)) if i%NUM_STEPS == step]

    shap_values_2D_step = shap_values_2D[indice]

    x_test_2d_step = x_test_2d.iloc[indice]

    print("_______ time step {} ___________".format(step))

    shap.summary_plot(shap_values_2D_step, x_test_2d_step, plot_type="bar")

    shap.summary_plot(shap_values_2D_step, x_test_2d_step)

    print("\n")