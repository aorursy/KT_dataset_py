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
filename = '/df_see_press2_30s_20190422_h11m36.csv'

df = pd.read_csv("../input" +filename, sep=';',index_col=False, header=0);  

#Adapt Datetime values to matplotlib 

from datetime import  datetime

df['DateTime1']= df['DateTime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))



#Calcolate "Oil Pump Active Power" and Oil Pump Reactive Power"  (as difference of power  beetwwn press's and oven's energies)

df['p2P_133']  = df['p2T_133'] - df['p2O_133'] # Oil Pump Active Power

df['p2P_134']  = df['p2T_134'] - df['p2O_134'] # Oil Pump Reactive Power

df['p2P_222d']  = df['p2T_222d'] - df['p2O_222d'] # Oil Pump Reactive Power



print('df.shape: ', df.shape)



df.head()

df.info()
df.describe()
print('Min DateTime: ', df['DateTime1'].min())

print('Max DateTime: ', df['DateTime1'].max())
import matplotlib.pyplot  as plt

import matplotlib.dates  as mdates



years     = mdates.YearLocator()   # every year

months    = mdates.MonthLocator()  # every month

days      = mdates.DayLocator()

hours      = mdates.HourLocator()

majorFmt     = mdates.DateFormatter('%m/%d/%Y') # %H:%M:%S')

minorFmt     = mdates.DateFormatter('%H:%M') 



n_plots = 6

fig, ax = plt.subplots(n_plots,1)



fig.autofmt_xdate()

for i in range(n_plots):

    ax[i].xaxis.set_major_formatter(majorFmt)

    ax[i].xaxis.set_major_locator(days)

    ax[i].xaxis.set_minor_formatter(minorFmt)



ax[0].set_ylabel('Press [W]')

ax[1].set_ylabel('Oven [W]')

ax[2].set_ylabel('Oil Pump [W]')

ax[3].set_ylabel('Oil Pump [Var]')

ax[4].set_ylabel('Oil Pump [Wh]')

ax[5].set_ylabel('Number of cycles')

ax[5].set_xlabel('time')



s_   = slice(0,len(df),1)

x    =   mdates.date2num(df['DateTime1'])[s_]

y0   =   df['p2T_133'][s_]

y1   =   df['p2O_133'][s_] 

y2   =   df['p2P_133'][s_]

y3   =   df['p2P_134'][s_]

y4   =   df['p2P_222d'][s_]

y5   =   df['plc1_1107d']



ax[0].plot(x,y0)

ax[1].plot(x,y1)

ax[2].plot(x,y2)

ax[3].plot(x,y3)

ax[4].plot(x,y4)

ax[5].plot(x,y5)

fig.set_size_inches(20,15)

plt.show()
print(df.columns)

#plt.matshow(df[['p2T_133', 'p2O_133', 'p2T_222d', 'p2O_222d', 'plc1_1107d']].corr())

flds = ['p2T_133', 'p2T_134','p2O_133', 'p2O_134', 'p2P_133','p2P_134', 'plc1_1107d']

flds = ['p2T_222d', 'p2O_222d',  'p2P_222d', 'p2P_134','plc1_1107d']

corr= df[flds].corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)



def AI_generator( data_in,

                  data_out,              #data matrix

                  lookback,              #steps in the past

                  delay,                 #steps in the future

                  min_index, max_index,  #range of delimiters for validation  and test data

                  shuffle = False,      #if or not shuffle data timebased

                  batch_size = 128,      # number of samples per batch; samples = lookback//batch_Size

                  step = 1,              #periods in timesteps, at which you sample data

                  target_sum = True,     #sum of output data

              ):

    #Generator used to slice data in the AI model

    

    if max_index is None:

            max_index   =   len(data_in) -1

    

    i = min_index + lookback

    

    while 1:

        if shuffle: 

            rows = np.random.randint(min_index+lookback,  max_index, size = batch_size)

        else:

            if i+batch_size >= max_index:

                i = min_index + lookback

            rows = np.arange(i, min(i+batch_size, max_index))

            i += len(rows)

        

        samples = np.zeros((len(rows),

                            lookback // step,

                            data_in.shape[-1]))

        targets = np.zeros ((len(rows),))

        

        for j, _ in enumerate(rows):

            indices = range(rows[j] - lookback, rows[j], step)

            samples[j] = data_in[indices,:]

            

            #Two output options: actual number of cycles or summation of number of cycles

            if target_sum:

                targets[j] = sum([data_out[x] for x in range(rows[j]-lookback+delay, rows[j]+delay)])

                #print('sum_target %s %s: ', (j,targets[j]))

            else:

                targets[j] = data_out[rows[j] + delay]

                

        yield samples, targets

        

        

float_data_in  = df[['p2T_133', 'p2T_134','p2O_133', 'p2O_134', 'p2P_133','p2P_134']].values

float_data_out = df[['plc1_1107d']].values

print('float_data_in.shape: ',float_data_in.shape)

print('float_data_out.shape: ',float_data_out.shape)

print('float_data_in:\n ', float_data_in)

print('float_data_out:\n ', float_data_out)
#Dictionary of configuration parameters

config_batchData = {

                'lookback'    : 10,          #number of points back for each batch of data

                'step'        : 1,          #numbers of rows step for each sample. samples = lookback//steps

                'delay'       : 0,          #delay from input data to temperature (one day)

                'shufle'      : False,      #shufle input order

                'batch_size'  : 4,          #number of data batches

                'target_sum'  : True,      #summarize last lookback data

                'steps_epochs': 64,        #64

                'epochs_'     : 100,

                'train_rate'  : 0.8,

                'val_rate'    : 0.15,

}





lookback        = config_batchData['lookback']     #number of rows for each batch of data

step            = config_batchData['step']          #numbers of rows step for each sample. samples = lookback//steps

delay           = config_batchData['delay']          #delay from input data to temperature (one day)

batch_size      = config_batchData['batch_size']         #number of data batches

target_sum      = config_batchData['target_sum']     #summarize last column data

steps_epochs    = config_batchData['steps_epochs']  #64

epochs_         = config_batchData['epochs_']

shufle          = config_batchData['shufle']

############################

#Data generators

############################





train_rows      = int( float_data_in.shape[0] * config_batchData['train_rate'] )         #numbers of rows to train 

val_rows        = int( float_data_in.shape[0] * (config_batchData['val_rate'] ) )     #numbers of rows to validation 

test_rows       = int( float_data_in.shape[0] * (1-config_batchData['train_rate'] - config_batchData['val_rate'] )   )      #numbers of rows to test 



train_steps     = int(train_rows + 1 - lookback) 

val_steps       = int(val_rows + 1 - lookback)

test_steps       = int(test_rows + 1 -lookback)

print('train_steps: ', train_steps)

print('val_steps:   '  , val_steps)

print('test_steps:   '  , test_steps)

                      

val_start =     0

val_end   =     val_start + val_rows        

train_start =   val_end + 1

train_end   =   train_start + train_rows

test_start =   train_end + 1

test_end   =   test_start + test_rows



# data generation          

#training data

train_gen = AI_generator( float_data_in,float_data_out,

                       lookback     = lookback,

                       delay        = delay,

                       min_index    = train_start,

                       max_index    = train_end,

                       shuffle      = shufle,

                       step         = step,

                       batch_size   = batch_size,

                       target_sum   = target_sum,)



#validation  data

val_gen = AI_generator( float_data_in,float_data_out,

                       lookback     = lookback,

                       delay        = delay,

                       min_index    = val_start,

                       max_index    = val_end,

                       shuffle      = shufle,

                       step         = step,

                       batch_size   = batch_size,                       

                       target_sum   = target_sum,)



test_gen = AI_generator( float_data_in,float_data_out,

                       lookback     = lookback,

                       delay        = delay,

                       min_index    = val_start,

                       max_index    = val_end,

                       shuffle      = shufle,

                       step         = step,

                       batch_size   = batch_size,                       

                       target_sum   = target_sum,)

print(float_data_in.shape)

from keras import models, layers, Input

from keras.optimizers import RMSprop

model = models.Sequential()



if 1==1:

    #simple model with no activation function

    #no memory, 

    #shuffled data

    input_tensor = Input(shape=(lookback//step, float_data_in.shape[-1]))

    x = layers.Flatten()(input_tensor)

    x = layers.Dense(64,activation='relu')(x)

    x = layers.Dense(32,activation='relu')(x)

    output_tensor = layers.Dense(1)(x)

    model = models.Model(input_tensor, output_tensor)

    



if 1==0:

    model.add(layers.Conv1D(

                            32,3 ,

                            activation='relu',

                            input_shape = (None,  float_data_in.shape[-1])))



    

    model.add(layers.MaxPooling1D(3))

    model.add(layers.Conv1D(

                                32, 5,

                                activation='relu',

                                ))

    model.add(layers.MaxPooling1D(3))

    model.add(layers.GRU(32, 

                             dropout = 0.1,

                             recurrent_dropout=0.5,

                             ))                  

    model.add(layers.Dense(1))

    

model.summary()
print('config_batchData: \n', config_batchData)



model.compile(  optimizer = RMSprop(), #lr=1e-4

                loss = 'mse', #'sparse_categorical_crossentropy', #'mse' 

                metrics = ['mae']

              )  



print('steps_epochs: ', steps_epochs)

print('epochs_: ', epochs_)

print('val_gen: ', val_gen)

print('val_steps: ', val_steps)



history = model.fit_generator(train_gen,

                              steps_per_epoch = steps_epochs, 

                              epochs=   epochs_,

                              validation_data = val_gen,

                              validation_steps = val_steps,

                              )
import matplotlib.pyplot as plt

loss_train   = history.history['loss']

loss_val     = history.history['val_loss']

epochs       = range(1,len(loss_train) + 1)



plt.plot(epochs, loss_train, 'bo', label = 'Training loss')

#plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')

plt.plot(epochs, loss_val, 'b', label = 'Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.legend()

plt.show()



#save model

model_name = 'model_p2_1.h5'

model.save(model_name) 






