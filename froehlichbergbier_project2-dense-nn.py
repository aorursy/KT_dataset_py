%matplotlib inline

import os

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout, LeakyReLU

from keras.constraints import MaxNorm

%load_ext autoreload

%autoreload 2
DEFAULT_BATCH_SIZE=32

TARGET_DIR = "/kaggle/working/figures"

if not os.path.isdir(TARGET_DIR):

    os.mkdir(TARGET_DIR)
### plots.py



class Error_plot_CV:

    def __init__(self, xvalues, xlabel, err_train, err_test, model_name=''):

        self.xvalues = xvalues

        self.xlabel = xlabel

        self.err_train = err_train

        self.err_test = err_test

        self.model_name = model_name



def cross_validation_visualization_many_in_one(*error_plots, title="cross validation", target_dir=TARGET_DIR, plotfunction=plt.plot):

    """Visualization the curves of mse_tr and mse_te.

    Takes as arguments any number of objects, each containing the following attributes:

    xvalues, xlabel, err_train, err_test, and model_name

    """

    # Plot

    plt.figure()

    marker_list=['*', '.', '+', 'o', '<', '>', '^', 'v', 'p', 'h']

    while len(marker_list) < len(error_plots):

        marker_list *= 2 # Make marker_list twice as long by repeating it.

    for marker, ep in zip(marker_list, error_plots): # zip() stops once the end of either of its arguments is reached.

        plotfunction(ep.xvalues, ep.err_train, marker=marker, linestyle = '--', label='train error '+ ep.model_name)

        plotfunction(ep.xvalues, ep.err_test, marker=marker, label='validation error '+ ep.model_name)

    

    plt.xlabel(ep.xlabel)

    plt.ylabel("RMSE")

    plt.title(title)

    plt.legend()

    plt.grid(True)

    if not os.path.isdir(target_dir):

        os.mkdir(target_dir)

    plt.savefig(os.path.join(target_dir, title))

    plt.show()

    

def standardize_scale(*args):

    """ Standardize the feature matrix x by multipling by 10E9 -> values will be >0 and around 1.

    """

    #set scale manualy to 10e9

    scale = np.power(10,9)

    

    for x in args:

        yield x*scale
reduce_ = False

quarter = True

assert not (quarter and reduce_)

x_file_path = "../data-input/20191205/cloak_mat_cs64{}_2.csv".format("" if quarter else "full")

annotation = "_quarter_64" if quarter else ""

reduce_str = "_reduced" if reduce_ else ""
# Load data (each row corresponds to one sample)

x_train = np.loadtxt('../input/quarter/x_train{}.csv'.format(annotation + reduce_str), dtype=np.float64, delimiter=',')

x_test  = np.loadtxt('../input/quarter/x_test{}.csv'.format( annotation + reduce_str), dtype=np.float64, delimiter=',')



# Reshape x to recover its 2D content

side_length = 32 if reduce_ else 64

#x_train = x_train.reshape(x_train.shape[0], side_length, side_length, 1)

#x_test = x_test.reshape(x_test.shape[0], side_length, side_length, 1)

print(x_train.shape)

print(x_test.shape)



# Load labels:

y_train = np.loadtxt('../input/quarter/y_train.csv', dtype=np.float64, delimiter=',')

y_test = np.loadtxt('../input/quarter/y_test.csv', dtype=np.float64, delimiter=',')



# Transform the labels y so that min(y) == 0 and max(y) == 1. Importantly, y_train and y_test must be considered jointly.

y_train, y_test = standardize_scale(y_train, y_test)

print(y_train.shape)

print(y_test.shape)
normal8000 = np.random.normal(size=8000)
BINS = 500
plt.figure()

plt.hist(normal8000, bins=BINS)

plt.show()
plt.figure()

plt.hist(y_train, bins=BINS)

plt.savefig(os.path.join(TARGET_DIR, "y_train_histogram"))
plt.figure()

plt.plot(np.sort(y_train), np.linspace(0,1, num=y_train.shape[0]+1)[1:])

plt.savefig(os.path.join(TARGET_DIR, "y_train_cfd"))
def init_dense_model(layer_sizes, dropout=0.2, max_norm_value=None):

    """Given a list of layer sizes, returns a compiled dense sequential neural network with 20% dropout.

    If the last element of layer_sizes is not 1, a 1 is appended to layer_sizes.

    Does not apply the constraint that the norm of the weight vector for each layer should be bounded above by 3.

    """

    if max_norm_value is not None:

        def myDense(*args, **kwargs):

            return Dense(*args, kernel_constraint=MaxNorm(max_value=max_norm_value), **kwargs)

    else:

        def myDense(*args, **kwargs):

            return Dense(*args, **kwargs)

        

    model = Sequential()

    if layer_sizes[-1] == 1 and len(layer_sizes) > 1:

        # Remove the last item

        layer_sizes = layer_sizes[:-1]

        

    if dropout:

        model.add(Dropout(dropout, input_shape=(4096,)))

        model.add(Dense(layer_sizes[0], kernel_constraint=MaxNorm(5.)))

        model.add(LeakyReLU())

    else:

        model.add(Dense(layer_sizes[0], input_shape=(4096,), kernel_constraint=MaxNorm(5)))

        model.add(LeakyReLU())

                  

    for size in layer_sizes[1:]:

        model.add(myDense(size))#, kernel_constraint=maxnorm(3)))

        model.add(LeakyReLU())

    

    model.add(myDense(1))

    model.add(LeakyReLU())

    return model
from keras.optimizers import SGD, Adam



def compile_and_fit (model, number_epochs, x=None, y=None, learning_rate=None, momentum=0.8, decay_rate=None):

    """ 

    Apply a learning rate schedule to the model based on the specified learning rate, momentum and number of epochs.

    The learning rate is decaying at each new epochs.

    After applying the learning rate, the model is compiled with the new 'SGD' and the chosen loss and metric.

    If fit=True: the model is fited to the specified x and y.

    """

    if (x is None and y is not None) or (x is not None and y is None):

        raise ValueError("Either specify x and y, or neither of them. You cannot just specify one and not the other. What kind of behaviour did you expect?")

    

#learning rate stuff:

#     sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

    #sgd = SGD(lr=learning_rate)

    if learning_rate is None:

        adam = Adam()

    elif decay_rate is None:

        adam = Adam(lr=learning_rate)

    else:

        adam = Adam(lr=learning_rate, decay=decay_rate)

#compiling

    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error', 'mae'])

    

#model fitting

    if (x is not None) and (y is not None):

        history = model.fit(x, y, epochs=number_epochs, validation_split=0.2, batch_size=DEFAULT_BATCH_SIZE)

    

    return model, history
class Architecture_NN:

    modes = ["pow"] # ["const", "ratio" , "pow", "exp"]

    def compute_layers(NN_depth, initial_layer, mode):

        layers = np.arange(0, NN_depth, dtype='int32')

        if mode == 'ratio':

            layers = np.ceil(initial_layer * (NN_depth - layers)/ NN_depth).astype('int32');

        elif mode == 'pow':

            layers = np.ceil(np.power(initial_layer, 1 / (layers+1))).astype('int32')

        elif mode == 'exp':

            layers = np.ceil(initial_layer * np.exp(-layers)).astype('int32')

        elif mode == 'const':

            layers = np.ones_like(layers) * initial_layer

        else:

            raise ValueError("Undefined mode: {}".format(mode))

        # Make sure all layers are >=2 and each layer is larger than the next

        print("computed layer sizes: ", layers)

        layers = np.maximum(layers, range(len(layers)+1, 1, -1))

        print("corrected layer sizes: ", layers)

        return layers
def grid_search_layer_size(NN_depth, initial_layer_sizes, mode, epochs, dropout=0.25, learning_rate=None, maxnorm=None, decay_rate=None):

    n_parameters, train_metrics, test_metrics = [], [], []

    

    for initial_layer in initial_layer_sizes:

        print("\n", '-'*10, NN_depth, "layers, ", initial_layer, " nodes in initial layer (" + mode + " mode)\n")

        model = init_dense_model(Architecture_NN.compute_layers(NN_depth, initial_layer, mode), dropout=dropout, max_norm_value=maxnorm)



        print(model.summary())

        model, history = compile_and_fit(model, epochs, x_train, y_train, learning_rate, decay_rate=decay_rate)

        metric_train = history.history['mean_squared_error'][-1]

        metric_test = history.history['val_mean_squared_error'][-1]

        print(model.summary())

        print("\n\nTrain error: {}\nTest error: {}\n\n".format(metric_train, metric_test))

        

        # Plot training & validation accuracy values

        plt.semilogy(np.sqrt(history.history['mean_squared_error']))

        plt.semilogy(np.sqrt(history.history['val_mean_squared_error']))

        plt.title('Model error')

        plt.ylabel('Root mean squared error')

        plt.xlabel('Epoch')

        plt.legend(['Train', 'Test'], loc='upper left')

        plt.savefig(os.path.join(TARGET_DIR, "Training_{}_hidden_layers__{}_mode__{}_neurons_in_layer1".format(NN_depth, mode, initial_layer)))

        plt.show()

            

        n_parameters.append(model.count_params())

        train_metrics.append(metric_train)

        test_metrics.append(metric_test)

    return n_parameters, train_metrics, test_metrics
def grid_search_mode_and_size(NN_depth, initial_layer_sizes, epochs=30, dropout=0.25, learning_rate=None, maxnorm=None, decay_rate=None):

    print("\n"*5, '='*25, NN_depth, "layers", '='*25, "\n\n")

    plot_args = []

    metric_train = None

    metric_test = None

    for mode in Architecture_NN.modes:

        n_parameters, train_metrics, test_metrics = grid_search_layer_size(NN_depth, initial_layer_sizes, mode, epochs, dropout=dropout, learning_rate=learning_rate, maxnorm=maxnorm, decay_rate=decay_rate)

        

        plot_args.append( Error_plot_CV(

                n_parameters, "Number of parameters", np.sqrt(train_metrics), np.sqrt(test_metrics), "({})".format(mode)

        ))

        

        if metric_test is None:

            metric_test = test_metrics[0]

            metric_train = train_metrics[0]

        

        ind = np.argmin(test_metrics)

        if test_metrics[ind] < metric_test:

            metric_test = test_metrics[ind]

            metric_train = train_metrics[ind]

    

    cross_validation_visualization_many_in_one(*plot_args, title="Tuning_{}_hidden_layers_{}_epochs".format(NN_depth, epochs))

    return metric_train, metric_test
import matplotlib.pyplot as plt

epochs = 50

NN_depths = [3] # range(1, 5)

initial_layer_sizes = np.power(2, np.arange(4, 11, 2))#[60] # np.power(2, np.arange(4, 9)) # 2^n from 16 to 256

dropout = 0.23

maxnorm = 0.5 # np.logspace(2, -2, num=20) #

learning_rate = 0.005 # np.logspace(0, -4, num=20)

decay_rates = [1e-3] #learning_rate / epochs * np.logspace(-2, 2, num=25)

train_metric_by_depth, test_metric_by_depth = [], []

for depth in NN_depths:

    for dr in decay_rates:

        metric_train, metric_test = grid_search_mode_and_size(depth, initial_layer_sizes, epochs=epochs, dropout=dropout, maxnorm=maxnorm, learning_rate=learning_rate, decay_rate=dr)

        train_metric_by_depth.append(metric_train)

        test_metric_by_depth.append(metric_test)

cross_validation_visualization_many_in_one(

        Error_plot_CV(

                decay_rates, "Decay rates (with learning rate = {})".format(learning_rate),

                np.sqrt(train_metric_by_depth), np.sqrt(test_metric_by_depth)

        ),

        title="Train_and_validation_error__dense_NN__tuned_layer_sizes", plotfunction=plt.semilogx

)
# cross_validation_visualization_many_in_one(

#         Error_plot_CV(

#                 learning_rates[:-2], "Learning rate",

#                 train_metric_by_depth[:-2], test_metric_by_depth[:-2]

#         ),

#         title="Train and validation error of dense NN", plotfunction=plt.plot# with tuned number of neurons per layer"

# )
import shutil

shutil.make_archive("/kaggle/working/output_figures_RMSE", 'zip', TARGET_DIR)