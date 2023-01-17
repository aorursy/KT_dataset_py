import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
from keras.layers import Input, Dense, BatchNormalization, GaussianNoise
from keras.models import Model
from keras.constraints import non_neg, MinMaxNorm, Constraint
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.metrics import mean_squared_error


def arsinh_helper_func(x):
	return np.log(x + np.sqrt(x**2 + 1))

def arsinh(x, scale=5):
	return arsinh_helper_func(x / scale)

try: # try: except in case I try to load the file again.
    data
except NameError:
    data = pd.read_csv('../input/Levine_32dim_notransform.csv')
    markers = list(data)[4:-5] # all channels that detect markers 
    data[markers] = arsinh(data[markers]) # Data transformation with the hyperbolic sine function
    data[data[markers] < 0] = 0 # Non-negativity!
data.head()
data.describe()
"""
    Testing regularization functions for the softmax layer
"""
def maximise_discrepancy_l1(R=0.01):
    assert 0 < R <= 1, "The rate must be between 0 and 1"
    def wrapped(A):
        """
        A is an activity vector, coming from a softmax activation.
        R is the regularization parameter and 
                            must be positive and less than than 1.
        For now it is set to 0.5
        The following function has a high value when 
        """
        sum_ = K.sum(A)
        # If sum_ > 1: raise: ValueError
        if K.greater(sum_, K.variable(1)) == K.variable(True):
            raise ValueError('Sum of A is greater than 1. Are you using softmax?')
        A_bar = K.mean(A)
        return ( 2 * (1 - A_bar) - (K.sum(K.abs(A-A_bar)))
            ) * R
    return wrapped
"""
(1 - (K.sum(K.abs(A - A_bar)) /
                     ((1 - A_bar)*2)
                     )
                ) * R"""

def maximise_discrepancy_exp(R):
    def wrapped(A):
        sum_ = K.sum(A)
        # If sum_ > 1: raise: ValueError
        if K.greater(sum_, K.variable(1)) == K.variable(True):
            raise ValueError('Sum of A is greater than 1. Are you using softmax?')
        sum_ = K.sum(A)
        A_bar = K.mean(A)
        return K.exp(-(K.sum(
                        K.abs(A - A_bar)
                            ) / (1-A_bar)*2
        )
    ) * R
    
    return wrapped

def maximise_discrepancy_l2(R=0.01):
    """
        A is an activity vector, coming from a softmax activation.
        R is the regularization parameter and 
                            must be positive and less than than 1.
    For now it is set to 0.5
    The following function has a high value when 
    """
    assert 0 < R <= 1 , "The rate must be between 0 and 1"
    def wrapped(A):
        sum_ = K.sum(A)
        A_bar = K.mean(A)
        return (1 - K.sqrt((K.sum((A - A_bar)**2)
                        ) / ((1 - A_bar) * 2))) * R
    return wrapped

def get_models(k, k_hidden, xshape, 
                params_hidden, params_clustering, params_out):
    """
        This function defines a model with the following architecture:
        BatchNorm(Input) -> BatchNorm(Hidden Layer) -> SoftMax(Decoder) ->
    """
    input_layer = Input(shape=(xshape,))
    input_layer_norm = GaussianNoise(0.05)(input_layer)
    input_layer_norm = BatchNormalization()(input_layer_norm)
    hidden_layer = Dense(k_hidden, **params_hidden)(input_layer_norm)
    hidden_layer = BatchNormalization()(hidden_layer)
    clustering = Dense(k, name='clustering',
                       **params_clustering)(hidden_layer)
    
    output = Dense(xshape, name='representation', 
                   **params_out)(clustering)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='mse', optimizer='nadam', metrics=['accuracy'])
    clustering_model = Model(inputs=input_layer, outputs=clustering)
    clustering_model.compile(loss='mse', optimizer='nadam', metrics=['accuracy'])
    return model, clustering_model

"""
This was the model before the custom constraint was added.

constraints = {'bias_constraint': non_neg(), 'kernel_constraint': non_neg()}
params_hidden = {'activation': 'sigmoid'}
params_clustering = {'activation': 'softmax',
                     'kernel_regularizer': regularizers.l1(1e-3),
                     'bias_regularizer': regularizers.l1(1e-3),
                    **constraints}

params_out = {**constraints, 'activation': 'relu', 
                    'bias_regularizer': regularizers.l2(1e-5),
                    'kernel_regularizer': regularizers.l2(1e-5)}

model, clustering_model = get_models(k, k_hidden, xshape,
                        #params_hidden, params_clustering, params_out)
"""
### Define a model with the custome regularization function
k = 30
k_hidden = 60 # experiment with this number
TRAIN_SIZE = 0.9

constraints = {'bias_constraint': non_neg(), 'kernel_constraint': non_neg()}
params_hidden = {'activation': 'sigmoid'}
xshape = data[markers].values.shape[1]
params_hidden = {'activation': 'sigmoid'}
params_clustering_2 = {'activation': 'softmax',
                        **constraints,
                      # 'kernel_regularizer': regularizers.l1(1e-3),
                       'activity_regularizer': regularizers.l1(1e-2)}
params_out_2 = {'activation': 'relu',
                **constraints,
              'kernel_regularizer': regularizers.l2(1e-3)}


model, clustering_model = get_models(k, k_hidden, xshape,
                                params_hidden, params_clustering_2, params_out_2)

earlyStopping = EarlyStopping(
                    monitor='loss',
                    patience=10,
                    verbose=0, mode='auto')

train_on = int(len(data[markers].values) * TRAIN_SIZE) # train size
model.fit(x=data[markers].values[:train_on], y=data[markers].values[:train_on],
        shuffle=True, batch_size=1024, verbose=1, epochs=200, callbacks=[earlyStopping]) 


print('Evaluation: ', model.evaluate(data[markers].values[train_on:], data[markers].values[train_on:]))
print('Clustering:?', clustering_model.predict(data[markers].values[0:2])[0])
### Evaluate MSE
print('Evaluation: ', model.evaluate(data[markers].values[train_on:], data[markers].values[train_on:]))
print('Evaluation: ', model.evaluate(data[markers].values[train_on:], data[markers].values[train_on:]))
clusterings = list(clustering_model.predict(data[markers].values[0:100]))
take = lambda x: clusterings[x]
clustering = take(1)
print('Clustering:?', clustering)

weights, biases = model.get_layer(name='representation').get_weights()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.bar(range(len(weights[0])), weights[np.argmax(clustering)])
ax1.set_title('weights')
ax2.bar(range(len(biases)), biases[np.argmax(clustering)])
ax2.set_title('biases')
ax3.bar(range(len(clustering)), clustering)
ax3.set_title('clustering')
plt.tight_layout()
plt.show()
### Assertions for non-negativity
get_lt0 = lambda x: sum(x < 0)
cl_predictions = clustering_model.predict(data[markers].values[train_on:]) #shape=(k,)
predictions = model.predict(data[markers].values[train_on:]) # shape=(34,)
assert sum(get_lt0(cl_predictions)) == 0 and sum(get_lt0(predictions)) == 0
(weights, biases), (cl_weights, cl_biases) = model.get_layer(name='representation').get_weights(),  model.get_layer(name='clustering').get_weights()
for i, _ in enumerate([weights, biases, cl_weights, cl_biases]):
    lt0_ = get_lt0(_)
    is_zero = np.all(lt0_ == 0)
    assert is_zero, (lt0_, i)

# output = Dense(xshape, name='representation', **params_)(clustering)
print('These are the weights for the layer that translates from hidden "code" to output.')
print('Thse weights should correspond to a combination of markers in each cluster.')
print('X axis corresponds to marker indices in data[markers]')
print('Y axis is the relative abundance/expression of the markers at the cluster.')

for i, weight in enumerate(weights):
    plt.bar(range(len(weight)), weight)
    plt.title('Example marker parameterization for cluster {}.'.format(i))
    plt.show()

print("""These are predictions made by the "clustering" step.
      These values corresponds to the clustering composition of each sample
      These should be sparse!.""")
print("""Now with the sparisty penalty it looks like this:""")
for i, cl_prediction in zip(range(20), cl_predictions):
    plt.bar(range(len(cl_prediction)), cl_prediction)
    plt.title('Softmax layer cluster assignment.')
    plt.show()