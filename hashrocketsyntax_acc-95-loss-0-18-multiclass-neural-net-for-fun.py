from datetime import datetime



import numpy as np

import pandas as pd





import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.layers.normalization import BatchNormalization

from keras.constraints import maxnorm

from keras.optimizers import Adam
time_start = datetime.now()



source = pd.read_csv("../input/sonar-data-set/sonar.all-data.csv")

sonar_data = source.sample(frac=1) #shuffle rows



x = sonar_data.drop("R", axis=1).values

scaler = StandardScaler() #normalize on mean of 0

x_scaled = scaler.fit_transform(x)



target = sonar_data['R']

y = pd.get_dummies(target, prefix='R').values # one hot encode for multi class



train_x, test_x, train_y, test_y = train_test_split(x_scaled, y, test_size=0.20) #0.30 wasn't enough data
model = Sequential(name='Sonar')



model.add(Dropout(0.20, input_shape=(60,))) # visible activations were pretty sparse (aka zeros), 

# ^ plus dropout is kind of like a premptive feature permutation.



model.add(Dense(60, 

                activation='relu', 

                kernel_initializer='he_uniform', 

                kernel_constraint=maxnorm(3))) # constraint w Normalize

model.add(Dense(30, 

                activation='relu', 

                kernel_initializer='he_uniform', 

                kernel_constraint=maxnorm(3))) # constraint w Normalize



model.add(BatchNormalization()) # single lava-hot neurons in final hidden layers sway results too much.



model.add(Dense(2, 

                activation='softmax', 

                kernel_initializer='glorot_uniform'))



opt = Adam(learning_rate=0.005) 

# ^ default settings getting stuck at 80% accurate, and I felt like it was a local min that could be rolled past w a little more umphhh.

model.compile(optimizer=opt, 

              loss='categorical_crossentropy', 

              metrics=['accuracy']) # again, just for fun.



print(model.summary())



history = model.fit(train_x, train_y, validation_data=(test_x, test_y), verbose = 0, batch_size=3, epochs=170)
plt.subplot(211)

plt.title('Loss')

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()



plt.subplot(212)

plt.title('Accuracy')

plt.plot(history.history['accuracy'], label='train')

plt.plot(history.history['val_accuracy'], label='test')

plt.legend()





plt.subplots_adjust(top=2, right=2)

plt.show()
results = model.evaluate(test_x, test_y)# (verbose = 0) won't print the epochs.



print(model.metrics_names)

print(results)
time_done = datetime.now()

print ("============================")

print ("\nStart : " + time_start.strftime("%Y-%m-%d %H:%M:%S") )

print ("Finish :" + time_done.strftime("%Y-%m-%d %H:%M:%S") )

print ("\n============================")



time_run = time_done - time_start

time_run_min = time_run.total_seconds() / 60

time_run_min_human = "{:.1f}".format(time_run_min)



print ("\nTotal runtime: " + time_run_min_human + " minutes.")

print (time_done.strftime("\n============================"))