# always start with checking out the files!
!ls ../input/*
import os
import math
import time

# Basic packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rd # generating random numbers
import datetime # manipulating date formats
# Viz
import matplotlib.pyplot as plt # basic plotting
import seaborn as sns # for prettier plots


# TIME SERIES
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

import datetime as dt

from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt


# settings
import warnings
warnings.filterwarnings("ignore")


# Import all of data inmediately  
sales=pd.read_csv("../input/sales_train.csv")

# settings
import warnings
warnings.filterwarnings("ignore")

item_cat=pd.read_csv("../input/item_categories.csv")
item=pd.read_csv("../input/items.csv")
sub=pd.read_csv("../input/sample_submission.csv")
shops=pd.read_csv("../input/shops.csv")
test=pd.read_csv("../input/test.csv")
#formatting the date column correctly
sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
# check
print(sales.info())
# number of items per cat 
#notice that the analisis tupla is simply x

x=item.groupby(['item_category_id']).count()

x=x.sort_values(by='item_id',ascending=False)
x=x.iloc[0:10].reset_index()
x
# #plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.item_category_id, x.item_id, alpha=0.8)
plt.title("Items per Category")
plt.ylabel('# of items', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()
sales.head()
#declaration of time-series kind of variable ts with groupby, but the structure is simple

ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()


ts.astype('float')
plt.figure(figsize=(16,8))
plt.title('Total Sales of the company')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts);
#with ts.rolling we are computing the mean and std of sales grouped in windows of size 12 in units of time (block times, indeed)

plt.figure(figsize=(16,6))
plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');
plt.plot(ts.rolling(window=12,center=False).std(),label='Rolling sd');
plt.legend();
#checking the requirements of the test file

test.head()
#Function that generate a dataframe with featured columns (total_cnt for instance)
    
def sales_summary_dataframe(shop_id_history, item_id_history):
   
    sales_shop_item = sales[(sales.shop_id==shop_id_history) & (sales.item_id == item_id_history)]
    monthly_sales_shop_item = sales_shop_item.groupby(["date_block_num"])["item_cnt_day"].sum()
    
    dates = pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
    
    nullhistory = pd.DataFrame({"node_tot_cnt":[0]*dates.shape[0]},columns=["node_tot_cnt"])
    
    if (sales_shop_item.shape[0] > 0):
        for con in range(monthly_sales_shop_item.shape[0]): 
            index_date_block = monthly_sales_shop_item.index[con]
            count_date_block = monthly_sales_shop_item.iloc[con]
   
            # here we fill the histories, always leaving the zero position to the featiure 
            # to be predicted [0]!!!!
            nullhistory.iloc[index_date_block,0] = count_date_block
    
    return nullhistory    
def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
class Timer():

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))
class Model():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
        
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def predict_point_by_point(self, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted
class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, dataframe, split, cols):
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x,y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                
                
                #normalised_col = [((float(p) / window[0,col_i]) - 1) for p in window[:, col_i]]
                
                #new normalization necessary since there are several times when window[0,col_i]=0
                #but still we like the idea to normalize each window wrt to the initial value
                
                normalised_col = [(float(p) - window[0,col_i]) for p in window[:, col_i]]
                
                normalised_window.append(normalised_col)
                
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
#we are going to use all the input file as train train_test_split": 1.0
#since we just need to predict the next month sales

#input_timesteps MUST BE EQUAL to sequence_length - 1
#in order to avoid array size compatibility problems

sequence_length = 6
input_timesteps = sequence_length - 1

configs = {"data": {"filename": "sales_train.csv", "columns": ["node_tot_cnt"],
                    "sequence_length": sequence_length, "train_test_split": 1.0, "normalise": True},
           "training": {"epochs": 2, "batch_size": 8},
           "model": {"loss": "mse", "optimizer": "adam", "save_dir": "saved_models",
                     "layers": [{"type": "lstm", "neurons": 100, "input_timesteps": input_timesteps, 
                                 "input_dim": 1, "return_seq": True},
                                {"type": "dropout", "rate": 0.2},
                                {"type": "lstm", "neurons": 100, "return_seq": True},
                                {"type": "lstm", "neurons": 100, "return_seq": False},
                                {"type": "dropout", "rate": 0.2},
                                {"type": "dense", "neurons": 1, "activation": "linear"}]}}
start_time=time.time()

if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

model = Model()
model.build_model(configs)
 
list_of_nodes =[]
list_of_predictions = []

#for node in range(len(test)):
for node in range(40):    
    
    # out-of memory generative training
    # here we should start the loop

    test_shop_id_index = test.iloc[node,1]
    test_item_id_index = test.iloc[node,2]
    
    sales_summary_df = sales_summary_dataframe(test_shop_id_index, test_item_id_index)
    data = DataLoader(sales_summary_df,
                  configs['data']['train_test_split'],
                  configs['data']['columns'])

    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    
    model.train_generator(
                data_gen=data.generate_train_batch(
                seq_len=configs['data']['sequence_length'],
                batch_size=configs['training']['batch_size'],
                normalise=configs['data']['normalise']
                ),
                epochs=configs['training']['epochs'],
                batch_size=configs['training']['batch_size'],
                steps_per_epoch=steps_per_epoch,
                save_dir=configs['model']['save_dir']
        )

     #making prediction based in the last block of months (for now we are using 5 months)

    ini_element = sales_summary_df.values.shape[0]-configs['data']['sequence_length'] + 1
    end_element = sales_summary_df.values.shape[0]
    last_train_block = np.array([sales_summary_df.values[ini_element:end_element]])

    #prediction for next month based on precious sequence_length months (last train block!!)
    memory_based_prediction_for_next_month = model.predict_point_by_point(last_train_block)
    
    list_of_nodes.append(node)
    list_of_predictions.append(memory_based_prediction_for_next_month[0])
    
    if (node % 10 == 0):
        end_time=time.time()
        print("forecasting for ",node,"th node and took",end_time-start_time,"s")
        start_time=end_time

#total submission file        
        
dfsubmission = pd.DataFrame({"ID":list_of_nodes,"item_cnt_month":list_of_predictions},
                            columns=["ID","item_cnt_month"])

dfsubmission.to_csv('submission_file.csv',index=False)
