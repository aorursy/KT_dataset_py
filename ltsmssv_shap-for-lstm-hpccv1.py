import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
print(os.listdir("../input/hpcc20steps/"))
!pip install tensorflow==1.14.0
variables_name = pd.read_csv("../input/hpcc20steps/variables_name.csv", header=None)

features = variables_name.values[:,1]
variables_name
features
import json

with open("../input/hpcc20steps/X_train_HPCC_1_20.json") as of:

    X_train = np.array(json.load(of))

with open("../input/hpcc20steps/y_train_HPCC_1_20.json") as of:

    y_train = np.array(json.load(of))

with open("../input/hpcc20steps/X_test_HPCC_1_20.json") as of:

    X_test = np.array(json.load(of))

with open("../input/hpcc20steps/y_test_HPCC_1_20.json") as of:

    y_test = np.array(json.load(of))    
X_train.shape, y_train.shape
X_test.shape, y_test.shape
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

model = createModel(8, 8, 8, 4, (X_train.shape[1], X_train.shape[2]))

model.fit(X_train, y_train, batch_size=8, epochs=30)
from sklearn.metrics import mean_squared_error as mse
y_pred_train = model.predict(X_train)

mse(y_train, y_pred_train)
y_pred = model.predict(X_test)

mse(y_test, y_pred)
model.summary()
# Save the entire model to a HDF5 file.

# The '.h5' extension indicates that the model shuold be saved to HDF5.

model.save('HPCCv1_model.h5') 
import shap
import tensorflow as tf

tf.__version__
# Use the training data for deep explainer => can use fewer instances

explainer = shap.DeepExplainer(model, X_train)

# explain the the testing instances (can use fewer instanaces)

# explaining each prediction requires 2 * background dataset size runs

shap_values = explainer.shap_values(X_test)

# init the JS visualization code

shap.initjs()
explainer.expected_value
len(shap_values)
X_test.shape
shap_values[0].shape
shap_values[0][0].shape
# shap.force_plot(explainer.expected_value[0], shap_values[0][0][0,:], features)

print(features)

print(len(features))
i=0

j=0
shap_values[0][i][j]
X_test[i][j].shape

# shap.force_plot(explainer.expected_value[0], shap_values[0][0], features)

i = 0

j = 0

x_test_df = pd.DataFrame(data=X_test[i][j].reshape(1,10), columns = features)

shap.force_plot(explainer.expected_value[0], shap_values[0][i][j], x_test_df)
shap.__version__
shap_values[0][i].shape
shap_values[0][0].shape
i = 11

pred_i = model.predict(X_test[i:i+1])

sum_shap_i = shap_values[0][i].sum() + explainer.expected_value[0]



pred_i, sum_shap_i
# Plot SHAP for ONLY one observation i

i = 0

shap.initjs()



x_test_df = pd.DataFrame(data=X_test[i], columns = features)

shap.force_plot(explainer.expected_value[0], shap_values[0][i], x_test_df)

## Problem:  Can not take into account many observations at the same time.

### The pic below explain for only 1 observation of 20 time steps, each time step has 10 features.
################# Plot AVERAGE shap values for ALL observations  #####################

## Consider ABSOLUTE of SHAP values ##

shap_average_value = np.abs(shap_values[0]).mean(axis=0)



x_average_value = pd.DataFrame(data=X_test.mean(axis=0), columns = features)

shap.force_plot(0, shap_average_value, x_average_value)
################# Plot AVERAGE shap values for ALL observations  #####################

## Consider average (+ is different from -)

shap_average_value = shap_values[0].mean(axis=0)



x_average_value = pd.DataFrame(data=X_test.mean(axis=0), columns = features)

shap.force_plot(explainer.expected_value[0], shap_average_value, x_average_value)
shap_values_2D = shap_values[0].reshape(-1,10)

X_test_2D = X_test.reshape(-1,10)





shap_values_2D.shape, X_test_2D.shape
x_test_2d = pd.DataFrame(data=X_test_2D, columns = features)
x_test_2d.corr()
shap.summary_plot(shap_values_2D, x_test_2d)
shap.summary_plot(shap_values_2D, x_test_2d, plot_type="bar")
len_test_set = X_test_2D.shape[0]

len_test_set
## SHAP for each time step

NUM_STEPS = 20

NUM_FEATURES = 10





# step = 0

for step in range(NUM_STEPS):

    indice = [i for i in list(range(len_test_set)) if i%NUM_STEPS == step]

    shap_values_2D_step = shap_values_2D[indice]

    x_test_2d_step = x_test_2d.iloc[indice]

    print("_______ time step {} ___________".format(step))

    shap.summary_plot(shap_values_2D_step, x_test_2d_step, plot_type="bar")

    shap.summary_plot(shap_values_2D_step, x_test_2d_step)

    print("\n")
# X_train_outlier

with open("../input/hpcc20steps/X_train_outlier.json") as of:

    X_train_outlier = np.array(json.load(of))

with open("../input/hpcc20steps/y_train_outlier.json") as of:

    y_train_outlier = np.array(json.load(of))



    # X_train_normal

with open("../input/hpcc20steps/X_train_not_outlier.json") as of:

    X_train_not_outlier = np.array(json.load(of))

with open("../input/hpcc20steps/y_train_not_outlier.json") as of:

    y_train_not_outlier = np.array(json.load(of))
## OUTLIERS

shap_values = explainer.shap_values(X_train_outlier)

i = 0

x_test_df = pd.DataFrame(data=X_train_outlier[i], columns = features)

shap.force_plot(explainer.expected_value[0], shap_values[0][i], x_test_df)
## NON-OUTLIERS

shap_values = explainer.shap_values(X_train_not_outlier)

i = 0

x_test_df = pd.DataFrame(data=X_train_not_outlier[i], columns = features)

shap.force_plot(explainer.expected_value[0], shap_values[0][i], x_test_df)
y_train_not_outlier[0]
y_train_outlier[0]
# Use the training data for deep explainer => can use fewer instances

explainer_2 = shap.GradientExplainer(model, X_train)

# explain the the testing instances (can use fewer instanaces)

# explaining each prediction requires 2 * background dataset size runs

shap_values_2 = explainer_2.shap_values(X_test)

# init the JS visualization code

shap.initjs()
################# Plot AVERAGE shap values for ALL observations  #####################

## Consider ABSOLUTE of SHAP values ##

shap_average_abs_value_2 = np.abs(shap_values_2[0]).mean(axis=0)



x_average_value = pd.DataFrame(data=X_test.mean(axis=0), columns = features)

shap.force_plot(0, shap_average_abs_value_2, x_average_value)
################# Plot AVERAGE shap values for ALL observations  #####################

## Consider ABSOLUTE of SHAP values ##

shap.initjs()

shap_values_train = explainer.shap_values(X_train)



shap_average_abs_value_train = np.abs(shap_values_train[0]).mean(axis=0)



x_average_value_train = pd.DataFrame(data=X_train.mean(axis=0), columns = features)

shap.force_plot(0, shap_average_abs_value_train, x_average_value_train)
shap_values_train_2D = shap_values_train[0].reshape(-1,10)

X_train_2D = X_train.reshape(-1,10)





shap.summary_plot(shap_values_train_2D, X_train_2D, features)
# COLOR: https://seaborn.pydata.org/tutorial/color_palettes.html

import seaborn as sns

import matplotlib.pyplot as plt



for i, feature in enumerate(features):

    print(feature)



    plt.figure(figsize = (8,6)) 

    tmp = shap_values_train[0][:,:,i].reshape((-1,20))

    print(tmp.shape)

    plot_shap = sns.heatmap(tmp, cmap="coolwarm")

    plt.show(plot_shap)

    print("-----------")