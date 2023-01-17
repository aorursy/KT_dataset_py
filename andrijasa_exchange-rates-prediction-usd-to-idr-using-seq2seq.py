from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import LSTM, Input, GRUCell, RNN
from keras.optimizers import Adam, SGD, RMSprop
from matplotlib import pyplot as plt
from numpy import array
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import CSVLogger
window_size =  40
df = pd.read_csv("../input/exrate2.csv")
df_norm = df.drop(['Date'], 1, inplace=True)
df_norm = df.drop(['Buy'], 1, inplace=True)

df = df.values[:]
df = np.array(df).astype(np.float32)
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)

df_train= df[:int(len(df)*0.8)]
df_val= df[int(len(df)*0.8):]
#%%

X = []
X_dec = []
y = []

for i in range(len(df_train) - window_size * 2):
    X.append(df_train[i:i + window_size])
    y.append(df_train[i + window_size:i + window_size * 2])
    #for dec_input=dec_output-1 teacher force
    temp = np.insert(df_train[i + window_size:i + (window_size * 2)-1],0,
                        #0)
                        df_train[i + window_size - 1:i + window_size])
    
    #for dec_input = zero
    #temp = np.zeros((window_size, 1))

    #for dec_input=enc_input-1
    #temp = np.insert(df_train[i:i + window_size-1],0,0)
    
    X_dec.append(temp)   

valX = []
valX_dec = []
valY = []
for i in range(len(df_val) - window_size * 2):
    valX.append(df_val[i:i + window_size])
    valY.append(df_val[i + window_size:i + window_size * 2])
    #for dec_input=dec_output-1
    temp = np.insert(df_val[i + window_size:i + (window_size * 2)-1],0,
                        #0)
                        df_val[i + window_size - 1:i + window_size])
    
    #for force teaching
    #temp = np.zeros((window_size, 1))

    #for dec_input=enc_input-1
    #temp = np.insert(df_val[i:i + window_size-1],0,0)
    
    valX_dec.append(temp)

#%%
X = np.array(X).astype(np.float32)
X_dec = np.array(X_dec).astype(np.float32)
#X = X.reshape(X.shape[0], 40, 1)
valX = np.array(valX).astype(np.float32)
valX_dec = np.array(valX_dec).astype(np.float32)
#valX = valX.reshape(valX.shape[0], 40, 1)
y = np.array(y).astype(np.float32)
valY = np.array(valY).astype(np.float32)

X = [X.reshape((X.shape[0],X.shape[1],1)), X_dec.reshape((X_dec.shape[0],X_dec.shape[1],1))]
y = y.reshape((y.shape[0],y.shape[1],1))
valX = [valX.reshape((valX.shape[0],valX.shape[1],1)), valX_dec.reshape((valX_dec.shape[0],valX_dec.shape[1],1))]
valY = valY.reshape((valY.shape[0],valY.shape[1]))

layers = [30,30]
dropout = 0.2
num_input_features = 1 
num_output_features = 1 

regulariser = None
input_sequence_length = 40 
target_sequence_length = 40 
num_steps_to_predict = 40 


encoder_inputs = Input(shape=(None, num_input_features), )

encoder_cells = []
for hidden_neurons in layers:
    encoder_cells.append(GRUCell(hidden_neurons, dropout=dropout,
                                              kernel_regularizer=regulariser,
                                              recurrent_regularizer=regulariser,
                                              bias_regularizer=regulariser))

encoder = RNN(encoder_cells, return_state=True)

encoder_outputs_and_states = encoder(encoder_inputs)

encoder_states = encoder_outputs_and_states[1:]

decoder_inputs = Input(shape=(None, 1))

decoder_cells = []
for hidden_neurons in layers:
    decoder_cells.append(GRUCell(hidden_neurons,
                                              kernel_regularizer=regulariser,
                                              recurrent_regularizer=regulariser,
                                              bias_regularizer=regulariser))

decoder = RNN(decoder_cells, return_sequences=True, return_state=True)

decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

decoder_outputs = decoder_outputs_and_states[0]

decoder_dense = Dense(num_output_features,
                                   activation='linear',
                                   kernel_regularizer=regulariser,
                                   bias_regularizer=regulariser)

decoder_outputs = decoder_dense(decoder_outputs)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

model.summary()
plot_model(model, to_file='encdec.png', show_shapes=True, show_layer_names=True)


batch_size = 128
epochs = 1000
learning_rate = 0.0001 #0.00001 adam
steps_per_epoch = None
decay = 0 # Learning rate decay
#optimiser = Adam(lr=learning_rate, decay=decay, amsgrad = False)
optimiser = RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)

# compile model
model.compile(loss='mse', optimizer=optimiser)

import datetime, time
import os
import sys    
file_name =  os.path.splitext(os.path.basename(sys.argv[0]))[0]
awal = datetime.datetime.now()
path = "{}_{}".format(file_name,awal.timestamp())
os.makedirs("{}".format(path))
filepath="{}/weights.best.hdf5".format(path)
csv_logger = CSVLogger('{}/training.log'.format(path))
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
monitor_earlydropping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto')
callbacks_list = [checkpoint, csv_logger , monitor_earlydropping]

# history = model.fit(X,y , epochs=epochs, 
#  			validation_data=(valX,valY ), batch_size=batch_size, shuffle=True, callbacks=callbacks_list)
history = model.fit(X,y , epochs=epochs, steps_per_epoch=steps_per_epoch, batch_size=batch_size,
			validation_split=0.2, shuffle=False, callbacks=callbacks_list)
# plot train and validation loss
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model train vs validation error \nlyr/nr={},lr={}, bs={}, opt={}".format(
            layers, learning_rate, batch_size, optimiser))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
#plt.show()
plt.savefig('{}/error.png'.format(path))

(x_encoder_test, x_decoder_test), y_test = valX, valY # x_decoder_test is composed of zeros.
y_test_predicted = model.predict([x_encoder_test, x_decoder_test])
predicted = scaler.inverse_transform(y_test_predicted.reshape((y_test_predicted.shape[0],
                                        y_test_predicted.shape[1])))
x_test = x_encoder_test.reshape((x_encoder_test.shape[0],x_encoder_test.shape[1]))
x_test = scaler.inverse_transform(x_test)
y_test = scaler.inverse_transform(y_test)

# Select 10 random examples to plot
indices = np.random.choice(range(x_test.shape[0]), replace=False, size=10)


for index in indices:
    plt.figure(figsize=(12, 5))

    past = x_test[index,:]
    true = y_test[index,:]
    pred = predicted[index,:]
    
    label1 = "Seen (past) values" 
    label2 = "True future values"
    label3 = "Predictions"

    plt.plot(range(len(past)), past, "o--b",
                label=label1)
    plt.plot(range(len(past),
                len(true)+len(past)), true, "x--b", label=label2)
    plt.plot(range(len(past), len(pred)+len(past)), pred, "o--y",
                label=label3)
    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    plt.show()
    #plt.show()
    plt.savefig('{}/graphpredict{}.png'.format(path,index))

#INFERENCE ENCODER AND DECODER
encoder_predict_model = Model(encoder_inputs,
                                           encoder_states)

decoder_states_inputs = []

for hidden_neurons in layers[::-1]:
    decoder_states_inputs.append(Input(shape=(hidden_neurons,)))

decoder_outputs_and_states = decoder(
    decoder_inputs, initial_state=decoder_states_inputs)

decoder_outputs = decoder_outputs_and_states[0]
decoder_states = decoder_outputs_and_states[1:]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_predict_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
#model.summary()
#plot_model(model, to_file='encdec.png', show_shapes=True, show_layer_names=True)
def predict(x, x_dec, encoder_predict_model, decoder_predict_model, num_steps_to_predict):
    
    y_predicted = []
    y2_predicted = []

    states = encoder_predict_model.predict(x)

    if not isinstance(states, list):
        states = [states]

    #decoder_input = np.zeros((x.shape[0], 1))
    #decoder_input = output = np.zeros((x.shape[0],1,1))
    #decoder_input = x_dec[:,:2,0].shape(len(x_dec),1,1)
    target_ = x_dec.shape[1]
    x_dec =  np.concatenate((x_dec,
            np.zeros((x_dec.shape[0],(num_steps_to_predict-x_dec.shape[1]),1))), axis=1)
    
    for i in range(num_steps_to_predict):
        decoder_input = np.array(x_dec[:,i:i+1,0]).reshape(len(x_dec),1,1)  
        outputs_and_states = decoder_predict_model.predict(
       [decoder_input] + states , batch_size=batch_size)
        output = outputs_and_states[0]
        states = outputs_and_states[1:]
        
        if ((i>=(num_steps_to_predict-target_-1)) and (i<=num_steps_to_predict-2) 
                and (num_steps_to_predict>40) and (decoder_input != np.zeros((len(x_dec), 1, 1))).all()):

            for j in range(num_steps_to_predict-1-i):
                x_dec[:,i+1+j,0]=output[:,0,0]

        y_predicted.append(output)

    return np.concatenate(y_predicted, axis=1)

def plot_prediction(x, y_true, y_pred, graphnum, isTest, path):
    
    plt.figure(figsize=(12, 5))

    past = x
    true = y_true
    pred = y_pred

    label1 = "Seen (past) values" 
    label2 = "True future values"
    label3 = "Predictions"

    plt.plot(range(len(past)), past, "o--b",
                label=label1)
    plt.plot(range(len(past),
                len(true)+len(past)), true, "x--b", label=label2)
    plt.plot(range(len(past), len(pred)+len(past)), pred, "o--y",
                label=label3)
    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    #plt.show()

    if isTest:
        plt.savefig('{}/graphTest{}.png'.format(path,graphnum))
    else:
        plt.savefig('{}/graphTrain{}.png'.format(path,graphnum))


(x_test, x_dec_test), y_test = valX, valY

y_test_predicted = predict(x_test, x_dec_test, encoder_predict_model, 
                                decoder_predict_model, num_steps_to_predict)

predicted_val = scaler.inverse_transform(y_test_predicted.reshape((y_test_predicted.shape[0],
                                        y_test_predicted.shape[1])))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]))
x_test = scaler.inverse_transform(x_test)
y_test = scaler.inverse_transform(y_test)

indices = np.random.choice(range(x_test.shape[0]), replace=False, size=10)

allpred = list()#store predictions across random model initializations
for index in indices:
    plot_prediction(x_test[index, :], y_test[index, :], predicted_val[index, :], 
                    index, isTest=True, path=path)
    allpred.append(np.ndarray.tolist(predicted_val[index, :]))
(x_train, x_dec_test), y_train = X,y

y_train_predicted = predict(x_train, x_dec_test, encoder_predict_model, decoder_predict_model, num_steps_to_predict)

predicted = scaler.inverse_transform(y_train_predicted.reshape((y_train_predicted.shape[0],
                                y_train_predicted.shape[1])))
x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]))
x_train = scaler.inverse_transform(x_train)
y_train = y_train.reshape((y_train.shape[0],y_train.shape[1]))
y_train = scaler.inverse_transform(y_train)

indices = np.random.choice(range(x_train.shape[0]), replace=False, size=10)

allpred2 = list()#store predictions across random model initializations
for index in indices:
    plot_prediction(x_train[index, :], y_train[index, :], predicted[index, :], 
                        index, isTest=False,path=path)
    allpred2.append(np.ndarray.tolist(predicted[index, :]))

#####ARIMA START######
from statsmodels.tsa.arima_model import ARIMA

train = df_train 
test = df_val 

model_arima = ARIMA(train, order=(0,1,2)) # ARIMA with grid searched parameters
model_fit = model_arima.fit(disp=0)
output = model_fit.forecast(steps=len(test))
yhat = output[0]
    
test = test.reshape((test.shape[0],test.shape[1]))
test = scaler.inverse_transform(test)
yhat = scaler.inverse_transform(yhat.reshape(-1, 1))

# plot
plt.figure(figsize=(12, 6))
label1 = "True values" 
label2 = "ARIMA" 
plt.plot(test[len(test)-window_size:], label=label1)
plt.plot(yhat[len(yhat)-window_size:], color='red', label=label2)
plt.plot(predicted_val[len(predicted_val)-1,:], color='gold', label=label3)

plt.ylabel('IDR')
plt.xlabel('Days')
plt.legend(loc='best')
plt.title("Model Comparison of IDR prediction over %d days" % window_size)
#plt.show()

plt.savefig('{}/graph4.png'.format(path))
inderror = (test[len(test)-window_size:] - yhat[len(yhat)-window_size:])**2
cummulativesumar = np.cumsum(inderror)
inderror = (test[len(test)-window_size:] - predicted_val[predicted_val.shape[0]-1,:].reshape(-1, 1))**2
cummulativesumalldl = np.cumsum(inderror)

plt.figure(figsize=(12, 6))

label1 = "Seq2Seq model " 
label2 = "ARIMA model" 

plt.plot(cummulativesumalldl,color='gold', label=label1)
plt.plot(cummulativesumar, color='red', label=label2)
plt.legend(loc='best')
plt.ylabel('Error')
plt.xlabel('Days')
plt.title("Model Comparison of RMSE over %d days" % window_size)
plt.show()
plt.savefig('{}/graph5.png'.format(path))


