###

### LIBRARIES

###



import os

import pandas as pd

import numpy as np

import random as rn

import datetime as dt

import time 

import gc





# matplotlib

%matplotlib inline

from matplotlib import pyplot as plt

import matplotlib.dates as mdates

from matplotlib import colors 



#seaborn

import seaborn as sns



from scipy.optimize import curve_fit, fsolve

import tensorflow as tf



#########################################################################################

#In order to keep the results of deep learning models as much 

# reproducibles as possible

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = ""



# The below is necessary for starting Numpy generated random numbers

# in a well-defined initial state.

np.random.seed(42)



# The below is necessary for starting core Python generated random numbers

# in a well-defined state.

rn.seed(12345)



# Force TensorFlow to use single thread.

# Multiple threads are a potential source of non-reproducible results.

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,

                              inter_op_parallelism_threads=1)



# The below tf.random.set_seed() will make random number generation

# in the TensorFlow backend have a well-defined initial state.

# For further details, see:

tf.random.set_seed(1234)



sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)

##########################################################################################





# Deep learning library for LSTM

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Input

from keras.layers import Dense

from keras.layers import Bidirectional

from keras.layers import Dropout

from keras.layers import Concatenate

from keras.layers import Masking

from keras.models import Model

from keras.optimizers import Adam

from keras import initializers

from keras import backend as K



from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.preprocessing import MinMaxScaler



import itertools

from copy import deepcopy

###

### FUNCTIONS

###





### DATA PREPARATION 



def train_test_creation(timeseries, train_test_days_split = 9):



    """

    This functions split timeseries in input as train and test data:

    Input:

        - timeseries: could be univariate or multivariate.

        - train_test_days_split: days of test data.

    """     



    n = len(timeseries)

    train, test = timeseries[0:(n-train_test_days_split)], timeseries[(n-train_test_days_split):]

    

    return train, test



def scale(n_features, train, test = None):

    

    """

    Defines scale transform for data in input.

    """ 

    

    if test is not None:

        # fit scaler

        scaler = MinMaxScaler(feature_range=(0, 1))

        train = train.reshape(-1, n_features)

        test = test.reshape(-1, n_features)

        scaler = scaler.fit(train)

        # transform train

        train_scaled = scaler.transform(train)

        # transform test

        test_scaled = scaler.transform(test)

        return scaler, train_scaled, test_scaled

    else:

        # fit scaler

        scaler = MinMaxScaler(feature_range=(0, 1))

        train = train.reshape(-1, n_features)

        scaler = scaler.fit(train)

        # transform train

        train_scaled = scaler.transform(train)

        

        return scaler, train_scaled

    



def invert_scale(scaler, value):

    

    """

    Returns inverse scale transform for value in input.

    """     



    array = np.array(value)

    array = array.reshape(-1, 1)

    inverted = scaler.inverse_transform(array)

    return inverted

       



def split_univariate(sequences, n_steps):

    

    """

    Transform univariate sequence to supervised sequence for LSTM.

    """ 

    

    n = len(sequences)

    X, y = list(), list()

    

    for i in range(n - n_steps):

        aux_x = sequences[i:(i+n_steps)]

        aux_y = sequences[i+n_steps]

        

        X.append(aux_x)

        y.append(aux_y)

    

    return np.array(X), np.array(y)





def prepare_data(data, train_test_days_split, n_steps, n_features, Normalization = True):



    # creation of train and test set: small test set since we have little data in this case.

    train, test = train_test_creation(data, train_test_days_split=train_test_days_split)

    

    if train_test_days_split != 0:

        

        # we have to create data to feed LSTM... 

        if Normalization:



            # univariate time sequences

            if n_features == 1:

                

                # normalization to [-1,1]

                scaler, train_scaled, test_scaled = scale(n_features, train, test)

                X_train, y_train = split_univariate(train_scaled, n_steps)

                X_test, y_test = split_univariate(test_scaled, n_steps)

                # ... and then reshape into correct dimension for the network [samples, time_steps, n_features]

                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],n_features))

                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],n_features))



            return X_train, y_train, X_test, y_test, scaler





        else:



            if n_features == 1:

                

                X_train, y_train = split_univariate(train, n_steps)

                X_test, y_test = split_univariate(test, n_steps)

                # ... and then reshape into correct dimension for the network [samples, time_steps, n_features]

                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],n_features))

                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],n_features))              



            return X_train, y_train, X_test, y_test

    

    ### ONLY TRAINING

    else:

        # we have to create data to feed LSTM... 

        if Normalization:



            # univariate time sequences

            if n_features == 1:

                # normalization

                scaler, train_scaled = scale(n_features, train)

                X_train, y_train = split_univariate(train_scaled, n_steps)

                # ... and then reshape into correct dimension for the network [samples, time_steps, n_features]

                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],n_features))



            return X_train, y_train, scaler





        else:



            if n_features == 1:

                X_train, y_train = split_univariate(train, n_steps)

                # ... and then reshape into correct dimension for the network [samples, time_steps, n_features]

                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],n_features))



            return X_train, y_train       

        



### FORECASTING



def forecast_point_by_point_multiple_LSTM(model, X_test_1, X_test_2, scaler_1, scaler_2, n_steps, n_features, window):



    predictions_cases_new = list()

    predictions_fatalities_new = list()

    

    # start sequence for total cases

    start_sequence_1 = X_test_1[-n_steps:,]

    print(n_steps)

    print(scaler_1.inverse_transform(start_sequence_1))

    # start sequence for total fatalities

    start_sequence_2 = X_test_2[-n_steps:,]



    for i in range(window):

        

        # reshape for predictions

        start_sequence_1 = start_sequence_1.reshape(1,n_steps,1)

        start_sequence_2 = start_sequence_2.reshape(1,n_steps,1)

        

        # predictions

        pred_cases_scaled, pred_fatalities_scaled = model.predict([start_sequence_1,start_sequence_2])

        

        # update start sequences for next prediction: insert last prediction and slice it using n_steps

        start_sequence_1 = np.insert(start_sequence_1,n_steps,pred_cases_scaled)

        start_sequence_1 = start_sequence_1[-n_steps:]

        start_sequence_2 = np.insert(start_sequence_2,n_steps,pred_fatalities_scaled)

        start_sequence_2 = start_sequence_2[-n_steps:]

        

        # rescale predictions...

        pred_cases = scaler_1.inverse_transform(pred_cases_scaled)[0,0]

        pred_fatalities = scaler_2.inverse_transform(pred_fatalities_scaled)[0,0]

        

        # ... and append the results of predictions

        predictions_cases_new.append(pred_cases)

        predictions_fatalities_new.append(pred_fatalities)

    

    return np.array(predictions_cases_new), np.array(predictions_fatalities_new)





### PLOTTING



def plot_forecasting_LSTM(df, country, variable, predictions):

    

    aux = df[df['IDRegion']==country][variable]

    date = df[df['IDRegion']==country]['Date']

    FMT = '%Y-%m-%d'

    start_date_acquistion = dt.datetime.strptime(date.min(),FMT)

    forecast_window = len(predictions)

    end_date = dt.datetime.strptime(date.max(),FMT) + dt.timedelta(days=forecast_window + 1)



    delta = end_date - start_date_acquistion       # as timedelta

    times = list()

    for i in range(delta.days + 1):

        day = start_date_acquistion + dt.timedelta(days=i)

        times.append(day)



    fig, ax = plt.subplots(figsize = (8,6))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))



    ax.plot(times[0:len(aux)], aux, '-ko', label='True', color = 'red')

    ax.plot(times[len(aux):(len(aux) + len(predictions))], predictions, '-ko', label='Predicted', color = 'blue')

    # position legend

    ax.set_xlabel('Date')

    ax.set_ylabel(variable)



    ax.legend(loc='upper left')

    

    fig.autofmt_xdate()

    fig.suptitle('{}'.format(country))

    plt.show()



### MODEL DEPLOYMENT    





def get_default_configuration(structure = 'classic',

                               n_steps = 1,

                               n_steps_out = None, 

                               n_features = 1,

                               neurons = 1,

                               activation_hidden = 'relu',

                               activation_out = 'linear',

                               loss_cases = 'mse',

                               loss_fatalities = 'mse',

                               optimizer = 'adam',

                               metrics = 'RMSE',

                               epochs = 2):

    

    """

    Defines default network configuration for LSTM

    """ 



    defaults = {'structure': structure, 

                'n_steps': n_steps,

                'n_steps_out': n_steps_out, 

                'n_features': n_features,

                'neurons': neurons,

                'activation_hidden': activation_hidden, 

                'activation_out' : activation_out,

                'loss_cases': loss_cases,

                'loss_fatalities': loss_fatalities,

                'optimizer': optimizer, 

                'metrics': metrics, 

                'epochs': epochs}

  

    return defaults



def get_tuning_model(options):

    

    """

    Returns a dict with all possible configuration given by options in input.

    """     



    keys = options.keys()

    values = (options[key] for key in keys)

    tuning_model = [dict(zip(keys, combination)) for combination in itertools.product(*values)]



    return tuning_model





def run(cases, fatalities, train_test_days_split, param_dict, tuning_model = None, Validation = False):

    

    """

    This functions takes in input train and test data, along with the default configuration for the newtork and one dictionary:

    tuning model. 

    Tuning model is a dictionary with alla possible configurations to run.

    It works in the following way:

        - Take the default configuration and substitute the configuration in input from the dictionary.

        - run the model and save the metric.

    """

    

    list_of_models = list()

    histories = list()



    # Itero sul numero di configurazioni da testare

    for i in range(len(tuning_model)):

        param_dict_temp = deepcopy(param_dict)

        dict_aux = deepcopy(tuning_model[i])



        # modifico i parametri della configurazione di default

        for parameter, options in dict_aux.items():

            param_dict_temp[parameter] = options



        # valuto modello e salvo metrica di accuratezza

        n_steps = param_dict_temp['n_steps']

        n_features = param_dict_temp['n_features']

        

        # check if needs log transformation

        transformation = param_dict_temp['transformation']

        

        # verify if padding is needed...

        if transformation == 'Log':

            

            cases_log = np.log(cases[cases > 0])

            fatalities_log= np.log(fatalities[fatalities > 0])

            len_cases_log = len(cases_log)

            len_fatalities_log = len(fatalities_log)

            

            max_len = np.max([len_cases_log, len_fatalities_log])

            

            # .. padding to have same length of sequences

            Xpad = padding(max_len, cases_log, fatalities_log)

            cases_final = Xpad[:,0]

            fatalities_final = Xpad[:,1]

        

        elif len(cases)!= len(fatalities):



            len_cases = len(cases)

            len_fatalities = len(fatalities)

            max_len = np.max([len_cases, len_fatalities])



            # .. function for padding padding()

            Xpad = padding(max_len, cases, fatalities)

            cases_final = Xpad[:,0]

            fatalities_final = Xpad[:,1]

            

        else:

            

            cases_final = cases

            fatalities_final = fatalities

        

        # prepare data

        X_train_1, y_train_1, X_test_1, y_test_1, _ =  prepare_data(cases_final, train_test_days_split = train_test_days_split, n_steps = n_steps, 

                                    n_features = n_features, Normalization = True)

        X_train_2, y_train_2, X_test_2, y_test_2, _ =  prepare_data(fatalities_final, train_test_days_split = train_test_days_split, n_steps = n_steps, 

                                    n_features = n_features, Normalization = True)

        

        # update metrics

        tuning_model[i]['metric'], model, history = evaluation(X_train_1, y_train_1, X_test_1, y_test_1,

                                                      X_train_2, y_train_2, X_test_2, y_test_2,

                                                      param_dict_temp, Validation = Validation)



        list_of_models.append(model)

        histories.append(history)

        K.clear_session()



    return pd.DataFrame(tuning_model), list_of_models, histories





    

def evaluation(X_train_1, y_train_1, X_test_1, y_test_1, X_train_2, y_train_2, X_test_2, y_test_2, param_dict, Validation = False):

    

    """

    Builds a LSTM model using the given params.

    """

    

    aux = multiple_LSTM(structure = param_dict['structure'],

                        n_steps = param_dict['n_steps'],

                        n_features = param_dict['n_features'],

                        neurons = param_dict['neurons'], 

                        activation_hidden = param_dict['activation_hidden'],

                        activation_out = param_dict['activation_out'],

                        loss_cases = param_dict['loss_cases'],

                        loss_fatalities = param_dict['loss_fatalities'],

                        optimizer = param_dict['optimizer'],

                        metrics = param_dict['metrics'],

                        epochs = param_dict['epochs'])

    

    aux.model_compile()

    metric, history = aux.evaluate_model(X_train_1, y_train_1, X_test_1, y_test_1,X_train_2, y_train_2, X_test_2, y_test_2, Validation = Validation)

    

    ### save txt of the results

    

    return metric, aux, history



def overall_fine_tuning(data, default_model_configuration, options, train_test_days_split, forecasting_window,

                          Public = True, Validation = False):

    

    """

    Fine tuning over a set of options and predictions with the best model found.

    Input:

        - data: train dataset.

        - default_model_configuration: model with default parameters.

        - options: dictionary with parameters for tuning.

        - end_training_date: date to split dataset between training and validation

        - transform: possible transformation to log (if so, padding is needed for sequences of cases and fatalities)

        - train_test_days_split: derives from end_training_date.

        - window: forecasting window

    Output:

        - list_of_countries: list of all countries analyzed.

        - predictions_cases_new: 

        - predictions_fatalities_new:

    """

    

    print("### STARTING FINE TUNING ###")

    

    # countries

    countries = np.unique(data['IDRegion'])

    

    # dictionary of parameters for fine tuning

    tuning_model = get_tuning_model(options)



    # lists in output

    list_of_countries = list()

    predictions_cases_new = list()

    predictions_fatalities_new = list()  



    count = 1

    total_time = 0

    

    for country in countries:

        

        print("--- {}) Starting with {}... ---".format(count, country))

        

        t0 = time.time()

        

        # variables for LSTM network

        cases = data[data['IDRegion']==country]['ConfirmedCases']

        fatalities = data[data['IDRegion']==country]['Fatalities']

        cases = np.array(cases)

        fatalities = np.array(fatalities)

        

        # if there's too much zero in the sequences, splitting training also in validation could

        # create more problem of convergence.

        len_cases_greater_than_zero = len(cases[cases > 0])       

        perc = round(len_cases_greater_than_zero/len(cases),2)

        if Validation:

            if (perc < 0.6):

                Validation = False

        

        # run of fine tuning

        df_results, list_of_models, list_of_histories = run(cases, fatalities, train_test_days_split,

                                         default_model_configuration,

                                         tuning_model=tuning_model,

                                         Validation = Validation)

        

        # extract the index of top model

        df_results = df_results.sort_values('metric')

        index = df_results.index[df_results['metric'] == df_results['metric'].min()].tolist()

        top_model = list_of_models[index[0]]

        

        

        n_steps = df_results['n_steps'].values[0]

        transformation = df_results['transformation'].values[0]



        predictions_cases, predictions_fatalities = single_predictions(top_model = top_model,

                                                                                     data = data,

                                                                                     country = country,

                                                                                     train_test_days_split = train_test_days_split,

                                                                                     n_steps = n_steps,

                                                                                     n_features = 1,

                                                                                     window = forecasting_window,

                                                                                     transform = transformation,

                                                                                     Public = Public,

                                                                                     Normalization = True)   

        # collect the predictions

        list_of_countries.append(country)

        predictions_cases_new.append(predictions_cases)

        predictions_fatalities_new.append(predictions_fatalities)



        

        print("--- {} in {} seconds ---".format(country,round((time.time() - t0))))

        

        count = count + 1

        total_time = total_time + round((time.time() - t0))

        gc.collect()

    

    print('### FINE TUNING LASTS {} MINUTES'.format(round(total_time/60)))

    

    return list_of_countries, predictions_cases_new, predictions_fatalities_new





def single_predictions(top_model, data, country, train_test_days_split, n_steps, n_features,  window, transform = None, 

                         Public = True, Normalization = True):

    

    cases = data[data['IDRegion']==country]['ConfirmedCases']

    fatalities = data[data['IDRegion']==country]['Fatalities']

    cases = np.array(cases)

    fatalities = np.array(fatalities)

    train_cases, test_cases = train_test_creation(cases, train_test_days_split=train_test_days_split)

    train_fatalities, test_fatalities = train_test_creation(fatalities, train_test_days_split=train_test_days_split)





    # transformation to log...

    if transform == 'Log':



        cases_log = np.log(cases[cases > 0])

        fatalities_log= np.log(fatalities[fatalities > 0])

        len_cases_log = len(cases_log)

        len_fatalities_log = len(fatalities_log)



        max_len = np.max([len_cases_log, len_fatalities_log])



        # .. function for padding padding()

        Xpad = padding(max_len, cases_log, fatalities_log)

        cases = Xpad[:,0]

        fatalities = Xpad[:,1]

        

    elif len(cases)!= len(fatalities):

        

        len_cases = len(cases)

        len_fatalities = len(fatalities)

        max_len = np.max([len_cases, len_fatalities])



        # .. function for padding padding()

        Xpad = padding(max_len, cases, fatalities)

        cases = Xpad[:,0]

        fatalities = Xpad[:,1]



    scaler_1, train_cases_scaled, test_cases_scaled = scale(1,train_cases, test_cases)

    scaler_2, train_fatalities_scaled, test_fatalities_scaled = scale(1,train_fatalities, test_fatalities)        



    # new predictions

    if Public:

        predictions_cases_new, predictions_fatalities_new = forecast_point_by_point_multiple_LSTM(top_model.model, train_cases_scaled, 

                                                                                                  train_fatalities_scaled,

                                                                                                  scaler_1, scaler_2,

                                                                                                  n_steps, n_features, window)

    else: 

        predictions_cases_new, predictions_fatalities_new = forecast_point_by_point_multiple_LSTM(top_model.model, test_cases_scaled, 

                                                                                                  test_fatalities_scaled,

                                                                                                  scaler_1, scaler_2,

                                                                                                  n_steps, n_features, window)

        

        

    return predictions_cases_new, predictions_fatalities_new



def padding(max_len, cases, fatalities):

    

    Xpad = np.full((max_len, 2), fill_value=-10, dtype = 'float')

    n_cases = len(cases)

    n_fatalities = len(fatalities)

    Xpad[-n_cases:,0] = cases

    Xpad[-n_fatalities:,1] = fatalities

    

    return Xpad





###

### CLASSES

###



# LSTM Class with in input only time sequences that need to be predicted: i.e. total cases and fatalities

class multiple_LSTM():

    

    """ 

    PARAMETERS:

        structure: string, which LSTM structure to use

        n_steps: number of time steps used

        n_steps_out: number of prediction time steps

        n_features: number of features in input (1 if just univariate time series)

        num_layers: int, number of hidden layers (excluding the input layer) 

        neurons: array of units/nodes in each layer

        activation: str, activation function in all layers except output

        loss: str, loss function

        optimizer: str, optimizer

        metrics: list of strings, metrics used

        epochs: int, number of epochs to train for

        batch_size: int, number of samples per batch

    """

    

    def __init__(self, structure, n_steps, n_features, 

                 neurons, activation_hidden, activation_out, loss_cases, loss_fatalities, optimizer, metrics,

                 epochs):

        

        self.structure = structure

        self.n_steps = n_steps

        self.n_features = n_features

        self.neurons = neurons

        self.activation_hidden = activation_hidden

        self.activation_out = activation_out

        self.loss_cases = loss_cases

        self.loss_fatalities = loss_fatalities

        self.optimizer = optimizer

        self.metrics = metrics

        self.epochs = epochs

        

        

    def model_compile(self):

        

        # call specific architecture base on kind of LSTM

        if self.structure == 'classic':

            self.classic_lstm()

    

        if self.structure == 'bidirectional':

            self.bidirectional_lstm()

            

            

    def evaluate_model(self, X_train_1, y_train_1, X_test_1, y_test_1, X_train_2, y_train_2, X_test_2, y_test_2, Validation = False):

        

        # Fit the model: fixed approach, fit once on the training data and then predict

        # each new time step one at a time from the test data.

        

        if Validation:

            history_callback = self.model.fit([X_train_1,X_train_2], [y_train_1,y_train_2],\

            epochs = self.epochs, validation_split = 0.1,

                                              verbose = 0, shuffle = False)

        else: 

            history_callback = self.model.fit([X_train_1,X_train_2], [y_train_1,y_train_2],\

            epochs = self.epochs, validation_data=([X_test_1, X_test_2],  [y_test_1, y_test_2]),

                                              verbose = 0, shuffle = False)



        # evaluate on test data.

        if self.metrics == 'RMSE':

            y_pred_cases, y_pred_fatalities = self.model.predict([X_test_1,X_test_2])

            pred = np.array([y_pred_cases,y_pred_fatalities]).reshape(-1)

            actual = np.array([y_test_1,y_test_2]).reshape(-1)

            metric= np.sqrt(mean_squared_error(pred, actual))

        elif self.metrics == 'RMSLE':

            y_pred_cases, y_pred_fatalities = self.model.predict([X_test_1,X_test_2])

            pred = np.array([y_pred_cases,y_pred_fatalities]).reshape(-1)

            actual = np.array([y_test_1,y_test_2]).reshape(-1)

            metric= np.sqrt(mean_squared_log_error(pred, actual,))                

            

            

        return metric, history_callback

        

            

    def classic_lstm(self):

      

        input_cases = Input(shape=(self.n_steps, self.n_features))

        input_fatalities = Input(shape=(self.n_steps, self.n_features))

        

        #LSTM cases

        MASKING_1_cases = Masking(mask_value=-10)(input_cases) # special value negative since with log transform in this case we take from >= 1

        LSTM_1_cases = LSTM(self.neurons)(MASKING_1_cases)

        DENSE_1_cases = Dense(self.neurons)(LSTM_1_cases)

        DROPOUT_1_cases = Dropout(0.3)(DENSE_1_cases)

        DENSE_final_cases = Dense(1, activation=self.activation_out, name = 'Cases')(DROPOUT_1_cases)

        

        #LSTM fatalities

        MASKING_1_fatalities = Masking(mask_value=-10)(input_fatalities) # special value negative since with log transform in this case we take from >= 1

        LSTM_1_fatalities = LSTM(self.neurons)(MASKING_1_fatalities)

        DENSE_1_fatalities = Dense(self.neurons)(LSTM_1_fatalities)

        DROPOUT_1_fatalities = Dropout(0.3)(DENSE_1_fatalities)

        DENSE_final_fatalities = Dense(1, activation=self.activation_out, name = 'Fatalities')(DROPOUT_1_fatalities)

        

        self.model = Model(inputs=[input_cases, input_fatalities], outputs=[DENSE_final_cases,DENSE_final_fatalities])

        

        optimizer = Adam(

        learning_rate=0.001)

        

        self.model.compile(loss=[self.loss_cases,self.loss_fatalities], 

                           optimizer=optimizer)

        

    def bidirectional_lstm(self):

      

        input_cases = Input(shape=(self.n_steps, self.n_features))

        input_fatalities = Input(shape=(self.n_steps, self.n_features))

        

        #LSTM cases

        MASKING_1_cases = Masking(mask_value=-10)(input_fatalities) # special value negative since with log transform in this case we take from >= 1

        LSTM_1_cases = Bidirectional(LSTM(self.neurons))(MASKING_1_cases)

        DENSE_1_cases = Dense(self.neurons)(LSTM_1_cases)

        # DROPOUT_1_cases = Dropout(0.3)(DENSE_1_cases)

        DENSE_final_cases = Dense(1, activation=self.activation_out, name = 'Cases')(DENSE_1_cases)

        

        #LSTM fatalities

        MASKING_1_fatalities = Masking(mask_value=-10)(input_fatalities) # special value negative since with log transform in this case we take from >= 1

        LSTM_1_fatalities = Bidirectional(LSTM(self.neurons))(MASKING_1_fatalities)

        DENSE_1_fatalities = Dense(self.neurons)(LSTM_1_fatalities)

        # DROPOUT_1_fatalities = Dropout(0.3)(DENSE_1_fatalities)

        DENSE_final_fatalities = Dense(1, activation=self.activation_out, name = 'Fatalities')(DENSE_1_fatalities)

        

        self.model = Model(inputs=[input_cases, input_fatalities], outputs=[DENSE_final_cases,DENSE_final_fatalities])

        

        optimizer = Adam(

        learning_rate=0.001)

        

        self.model.compile(loss=[self.loss_cases,self.loss_fatalities], 

                           optimizer=optimizer)

        

        

# LSTM Class with in input time sequences that need to be predicted and also auxiliary data

###

### LOADING DATA

###



filename_train = '../input/covid19-global-forecasting-week-4/train.csv'

filename_test = '../input/covid19-global-forecasting-week-4/test.csv'

filename_submission = '../input/covid19-global-forecasting-week-4/submission.csv'



filename_enriched = '../input/covid-19-enriched-dataset-week-2/enriched_covid_19_week_2.csv'



### 1) Covid_Data: 

data_train = pd.read_csv(filename_train)



data_train['Province_State'] = data_train['Province_State'].fillna('Nation')

data_train['IDRegion'] = data_train['Country_Region'] + ' ' + data_train['Province_State']



# data_train['ConfirmedCases'] = data_train['ConfirmedCases'].astype('float32')

# data_train['Fatalities'] = data_train['Fatalities'].astype('float32')



data_test = pd.read_csv(filename_test)



data_test['Province_State'] = data_test['Province_State'].fillna('Nation')

data_test['IDRegion'] = data_test['Country_Region'] + ' ' + data_test['Province_State']



data_submission = pd.read_csv(filename_submission)





# Enriched Data

data_enriched = pd.read_csv(filename_enriched)



data_enriched.head()
###

### TRAINING AND VALIDATION PERIOD: PUBLIC LEADERBOARD

###



total_length = data_train.shape[0]/len(np.unique(data_train['IDRegion']))



end_training_date_public = '2020-04-01'

public_leaderboard = '2020-04-15'

end_test_date = data_train['Date'].max()



FMT = '%Y-%m-%d'



diff = np.int((dt.datetime.strptime(end_test_date,FMT) - dt.datetime.strptime(end_training_date_public,FMT)).days)

public_leaderboard_window = np.int((dt.datetime.strptime(public_leaderboard,FMT) - dt.datetime.strptime(end_training_date_public,FMT)).days)
###

### FINE TUNING - PUBLIC

###



# For each country we run fine tuning algorithm to find the best lag parameter

options = {

    'transformation': ['Normal'],

    'structure': ['classic'],

    "neurons": [128],

    "n_steps": [1,2,3,4],

    "epochs": [200]

}



# other parameters...

default_model_configuration = get_default_configuration(structure = 'classic',

                        n_steps = 1, 

                        n_features = 1,

                        neurons = 20, 

                        activation_hidden = 'relu',

                        activation_out = 'linear',

                        loss_cases = tf.keras.losses.MeanSquaredLogarithmicError(),

                        loss_fatalities = tf.keras.losses.MeanSquaredLogarithmicError(), 

                        optimizer = 'adam',

                        metrics = 'RMSE',

                        epochs = 400)



# predictions

list_of_countries, predictions_cases_new, predictions_fatalities_new = overall_fine_tuning(data = data_train,

                                                                                           default_model_configuration = default_model_configuration, 

                                                                                           options = options,

                                                                                           train_test_days_split = diff,

                                                                                           forecasting_window = public_leaderboard_window,

                                                                                           Public = True,

                                                                                           Validation = False)



# relate predictions to country with a dict

zip_cases = zip(list_of_countries, predictions_cases_new)

zip_fatalities = zip(list_of_countries, predictions_fatalities_new)



# Create a dictionary from zip object

dict_of_predictions_cases_public = dict(zip_cases)

dict_of_predictions_fatalities_public = dict(zip_fatalities)

###

### PLOT FORECASTING CASES: PUBLIC

###



country = 'France Nation'



date_for_predictions_public = data_train[data_train['Date'] <= end_training_date_public]

variable = 'ConfirmedCases'

predictions_cases_public = dict_of_predictions_cases_public[country]

plot_forecasting_LSTM(date_for_predictions_public, country, variable, predictions_cases_public)
###

### PLOT FORECASTING FATALITIES: PUBLIC

###



variable = 'Fatalities'

predictions_fatalities_public = dict_of_predictions_fatalities_public[country]

plot_forecasting_LSTM(date_for_predictions_public, country, variable, predictions_fatalities_public)
###

### TRAINING AND VALIDATION PERIOD: PRIVATE LEADERBOARD

###



total_length = data_train.shape[0]/len(np.unique(data_train['IDRegion']))



end_training_date_private = data_train['Date'].max()

public_leaderboard = '2020-04-15'

private_leaderboard = '2020-05-14'



FMT = '%Y-%m-%d'



diff = np.int((dt.datetime.strptime(public_leaderboard,FMT) - dt.datetime.strptime(end_training_date_private,FMT)).days) - 1

private_leaderboard_window = np.int((dt.datetime.strptime(private_leaderboard,FMT) - dt.datetime.strptime(public_leaderboard,FMT)).days)

###

### FINE TUNING - PRIVATE

###



# For each country we run fine tuning algorithm to find the best lag parameter

options = {

    'transformation': ['Normal'],

    'structure': ['classic'],

    "neurons": [128],

    "n_steps": [1,2,3,4],

    "epochs": [200]

}



# other parameters...

default_model_configuration = get_default_configuration(structure = 'classic',

                        n_steps = 1, 

                        n_features = 1,

                        neurons = 20, 

                        activation_hidden = 'relu',

                        activation_out = 'linear',

                        loss_cases = tf.keras.losses.MeanSquaredLogarithmicError(),

                        loss_fatalities = tf.keras.losses.MeanSquaredLogarithmicError(), 

                        optimizer = 'adam',

                        metrics = 'RMSE',

                        epochs = 400)



# predictions

list_of_countries, predictions_cases_new, predictions_fatalities_new = overall_fine_tuning(data = data_train,

                                                                                           default_model_configuration = default_model_configuration, 

                                                                                           options = options,

                                                                                           train_test_days_split = 7,

                                                                                           forecasting_window = private_leaderboard_window + diff,

                                                                                           Public = False,

                                                                                           Validation = False)



# relate predictions to country with a dict

zip_cases = zip(list_of_countries, predictions_cases_new)

zip_fatalities = zip(list_of_countries, predictions_fatalities_new)



# Create a dictionary from zip object

dict_of_predictions_cases_private = dict(zip_cases)

dict_of_predictions_fatalities_private = dict(zip_fatalities)

###

### PLOT FORECASTING CASES: PRIVATE

###



country = 'Afghanistan Nation'



date_for_predictions_private = data_train[data_train['Date'] <= end_training_date_private]

variable = 'ConfirmedCases'

predictions_cases_private = dict_of_predictions_cases_private[country]

plot_forecasting_LSTM(date_for_predictions_private, country, variable, predictions_cases_private)
###

### PLOT FORECASTING FATALITIES: PRIVATE

###



date_for_fatalities_private = data_train[data_train['Date'] <= end_training_date_private]

variable = 'Fatalities'

predictions_fatalities_private = dict_of_predictions_fatalities_private[country]

plot_forecasting_LSTM(date_for_predictions_private, country, variable, predictions_fatalities_private)
###

### SUBMISSION FILE

###



### PUBLIC

# assing predictions to submission file

for key in dict_of_predictions_fatalities_public:

  

    # cases

    cases = dict_of_predictions_cases_public[key]

    

    # fatalities

    fatalities = dict_of_predictions_fatalities_public[key]

    

    # id

    fid = data_test[data_test['IDRegion']==key]['ForecastId'].values

    fid = fid[0:public_leaderboard_window]

    

    data_submission[data_submission.ForecastId.isin(fid)] = data_submission[data_submission.ForecastId.isin(fid)].assign(ConfirmedCases=cases)

    data_submission[data_submission.ForecastId.isin(fid)] = data_submission[data_submission.ForecastId.isin(fid)].assign(Fatalities=fatalities)





### PRIVATE



# # assing predictions to submission file

for key in dict_of_predictions_cases_private:

  

    # cases

    cases = dict_of_predictions_cases_private[key]

    cases = cases[diff:]

    

    # fatalities

    fatalities = dict_of_predictions_fatalities_private[key]

    fatalities = fatalities[diff:]

    

    # id

    fid = data_test[data_test['IDRegion']==key]['ForecastId'].values

    fid = fid[public_leaderboard_window:]

    

    data_submission[data_submission.ForecastId.isin(fid)] = data_submission[data_submission.ForecastId.isin(fid)].assign(ConfirmedCases=cases)

    data_submission[data_submission.ForecastId.isin(fid)] = data_submission[data_submission.ForecastId.isin(fid)].assign(Fatalities=fatalities)

        

data_submission.to_csv('submission.csv',index=False)
