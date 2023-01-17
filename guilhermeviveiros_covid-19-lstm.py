import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt
path_confirmed = "../input/covid-19-cssegisanddata/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

path_deaths = "../input/covid-19-cssegisanddata/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

path_recovered = "../input/covid-19-cssegisanddata/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
death_cases = pd.read_csv(path_deaths)

recovered_cases = pd.read_csv(path_recovered)



df = pd.read_csv(path_confirmed)

df.drop( labels = ["Province/State","Country/Region","Lat","Long"], inplace = True , axis = 1)

df
data = df.sum().to_frame(name='cases').to_numpy()

data
from sklearn.preprocessing import MinMaxScaler



#fit and transform or fit_transform

scaler = MinMaxScaler(feature_range=(-1, 1))

data = scaler.fit_transform(data)





print("data normalizada = " + str(data.shape))
plt.plot(data)

plt.title("Covid-19")

plt.xlabel("Days")

plt.ylabel("Confirmed Cases")

plt.show()
def makeXy(data,time_step,forecasting):

    

    X = []

    Y = []

    dataset = []

    

    i = 0

    step = time_step

    step_y = time_step

    

    if(forecasting == "single"):

        step_y = 1

        

        



    while((step+7) <= len(data)):

    

        for tmp in range(i,step):

            

            X.append(data[tmp])

        

       

        for tmp in range(step,step+step_y):

            

            Y.append(data[tmp])

            

        #Aqui um ciclo acaba

        dataset.append([X,Y])

        X = []

        Y = []

        

        i = i + 1

        step = step + 1

    

    #Agora colocar esta list em numpy array

   

    X = np.array([case[0] for case in dataset]).reshape(len(dataset),time_step)

    y = np.array([case[1] for case in dataset]).reshape(len(dataset),step_y)

        

    #Reshape para ser possível passar à LSTM

    X = X.reshape(len(X),len(X[0]),1)

    y = y.reshape(len(y),len(y[0]),1)

    

    return X,y



X,y = makeXy(data,time_step = 7, forecasting = "single")

#dataset
print(X[0:2])



print("\n-------------------\n")



print(y[0:1])
def show_history(history):

    print(history.history.keys())



    # summarize history for accuracy

    #plt.plot(history.history['mean_squared_error'])

    #plt.title('model accuracy with 128 units')

    #plt.ylabel('accuracy')

    #plt.xlabel('epoch')

    #plt.show()

    # summarize history for loss

    plt.plot(history.history['loss'],'r')

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.show()
def build_model(forecasting,time_steps,nr_features,units):

    

    #tf.keras.backend.clear_session()

    model = tf.keras.Sequential()

    

    model.add(tf.keras.layers.LSTM(units = units, input_shape= (time_steps, nr_features), return_sequences = False))

    

    if(forecasting == "multi"):

    

        #model.add(tf.keras.layers.LSTM(units = units, return_sequences = False))

    

        model.add(tf.keras.layers.Dense(64,activation='linear'))

        

        model.add(tf.keras.layers.Dense(time_steps,activation='linear'))    

        #model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1,activation='linear')))



    if(forecasting == "single"):

        

        #model.add(tf.keras.layers.LSTM(units = units, return_sequences = False))

    

        model.add(tf.keras.layers.Dense(64,activation='linear'))

        

        model.add(tf.keras.layers.Dense(1,activation='linear'))        

        

    opt = tf.keras.optimizers.Adam()

    

    model.compile(optimizer="adam",loss = "mse")# metrics=[tf.keras.metrics.MeanSquaredError()])

    

    return model





from keras.callbacks.callbacks import EarlyStopping



def fit(epochs,model,X,y):

    

    callbacks = [

        EarlyStopping(

            monitor='loss',

            patience=3,

            verbose=2,

            mode='min'

        )

    ]

    

    history = model.fit(X,y,shuffle=False,verbose=2,epochs=epochs)#callbacks=callbacks)

    

    return history
X,y = makeXy(data,7,"multi")

model_multi_step  = build_model("multi",7, 1, 128)

model_multi_step.summary()

history = fit(300,model_multi_step,X,y)
show_history(history)
X,y = makeXy(data,7,"single")

model_single_step  = build_model("single",7, 1, 128)

model_single_step.summary()

history = fit(100,model_single_step,X,y)
show_history(history)
def predict_data(data,model,forecasting,time_step):

    m = len(data)

    i = 0

    predicted_array = []

    

    #dou sempre append dos primeiros 7 valores, são os únicos que não são previstos

    for i in range(time_step):

        predicted_array.append(data[i])

    

    

    if(forecasting == "multi"):

        

        while(i+time_step < m):

            

            #da 7 valores

            predict = model.predict([[data[i:i+time_step]]])[0]

            

            for value in predict:

    

                predicted_array.append(value)



            i = i + time_step

    if(forecasting == "single"):

        

        while(i+time_step < m):

            

            #da 1 valor

            predict = model.predict([[data[i:i+time_step]]])[0]

            

            predicted_array.append(predict[0])



            i = i + 1

            

    

    return np.asarray(predicted_array)
def return_results(time_step,model1,model2):



    # Multi Step

    time_step = time_step



    predicted_x_train_multi = predict_data(data,model1,"multi",time_step) #preve nos dados de treino

    predicted_days_multi = model1.predict([[data[-time_step:]]])[0]#preve os 7 dias seguintes





    #Single Step

    predicted_x_train_single = predict_data(data,model2,"single",time_step) #preve nos dados de treino

    predicted_days_single = np.zeros(shape=(time_step,1))



    tmp = 0

    for i in data[-time_step:]:

        predicted_days_single[tmp][0] = i

        tmp = tmp + 1

    



    #preve os proximos 7 dias

    for i in range(time_step):

        value = model2.predict([[predicted_days_single]])[0]#preve os 7 dias seguintes

   

        for i in range(0,len(predicted_days_single)-1):

        

            predicted_days_single[i] = predicted_days_single[i+1]

        

        

        predicted_days_single[i+1] = value

        

    dt1 =  []

    dt2 = []



    for i in data:

        dt1.append(i[0])

        dt2.append(i[0])

    for i in predicted_days_multi:

        dt1.append(i)

    for i in predicted_days_single:

        dt2.append(i[0])

        

    return predicted_x_train_multi,dt1,predicted_x_train_single,dt2



predicted_x_train_multi,predicted_days_multi,predicted_x_train_single, predicted_days_single = return_results(7,model_multi_step,model_single_step)
from matplotlib.pyplot import figure



plt.figure(figsize=(15,10))

    

plt.plot(predicted_x_train_single,'k')

plt.plot(predicted_x_train_multi,'r')

plt.plot(predicted_days_multi,'b')

plt.plot(predicted_days_single,'c')

plt.plot(data,'g')



plt.title("Number of Confirmed cases with Covid-19")

plt.xlabel("Days")

plt.ylabel("Covid-19")

plt.legend(['Predicted on the same data using forecasting single step',

            'Predicted on the same data using forecasting multi step(7)',

            'Predicted on 7 days using multi step',

            'Predicted on 7 days using single step',

            'Real Data'],loc = 'upper left')

plt.legend

plt.show()
actual_value = scaler.inverse_transform(np.asarray(data[-1:]).reshape(1,-1))[0][0]

max_value1 = scaler.inverse_transform(np.asarray(predicted_days_single[-1:]).reshape(1,-1))[0][0]

max_value2 = scaler.inverse_transform(np.asarray(predicted_days_multi[-1:]).reshape(1,-1))[0][0]





print("Valor atual de casos confirmados = " + str(actual_value))

print("Valor atual de casos 7 dias depois segundo o model single step = " + str(max_value1))

print("Valor atual de casos 7 dias depois segundo o model multi step = " + str(max_value2))

X,y = makeXy(data,4,"multi")

model_ms  = build_model("multi",4, 1, 128)

model_ms.summary()

history = fit(100,model_ms,X,y)



X,y = makeXy(data,4,"single")

model_ss  = build_model("single",4, 1, 128)

model_ss.summary()

history = fit(100,model_ss,X,y)
predicted_x_train_multi,predicted_days_multi,predicted_x_train_single,predicted_days_single = return_results(4,model_ms,model_ss)
from matplotlib.pyplot import figure



plt.figure(figsize=(15,10))

    

plt.plot(predicted_x_train_single,'k')

plt.plot(predicted_x_train_multi,'r')

plt.plot(predicted_days_multi,'b')

plt.plot(predicted_days_single,'c')

plt.plot(data,'g')



plt.title("Number of Confirmed cases with Covid-19")

plt.xlabel("Days")

plt.ylabel("Covid-19")

plt.legend(['Predicted on the same data using forecasting single step',

            'Predicted on the same data using forecasting multi step(7)',

            'Predicted on 7 days using multi step',

            'Predicted on 7 days using single step',

            'Real Data'],loc = 'upper left')

plt.legend

plt.show()