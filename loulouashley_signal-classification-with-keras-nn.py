import numpy as np

import matplotlib.pyplot as plt

import csv

file_name = "Example1.csv"

DataInput = []

IndexMax = 10000

f = lambda x: 3*np.sin(3*x/IndexMax*2*np.pi)



for i in range(IndexMax) :

    DataInput.append(f(i))
# Export of the fonction in a folder 

out_file = open(file_name, 'w', newline='')

wrt = csv.writer(out_file)

wrt.writerow(DataInput)

out_file.close()



# Re-importation of the fonction and draw of it

in_file = open(file_name, 'r')

reader = list(csv.reader(in_file, quoting=csv.QUOTE_NONNUMERIC))

DataInput = reader[0]

in_file.close()
plt.plot(DataInput)

plt.show()
# LSTM pour une variable simple

from numpy import array

import math

import random

import matplotlib.pyplot as plt

import csv

import numpy as np

from keras.models import Sequential

from keras.layers import Flatten

from keras.layers import Dense
n_steps = 1 # number of neurones for the NN

n_features = 1 # reshape value 



file_name = "Example1.csv"



Data_output=np.zeros([10000,1]) # Initialisation of databases for further using

BDDtrain=np.zeros([2000,10000])

BDDytarget=np.zeros([2000,8])



#BDD des differents cas nominaux :



for i in range(250):               #Loop for the first 1000 nominal cases - category 1

    for j in range (10000):

        BDDtrain[i,j] = DataInput[j] + random.random()



for i in range(250):               #Loop for the  1000 nominal cases - category 2

    for j in range (10000):

        BDDtrain[i + 250 ,j] = 1.25 * DataInput[j] + random.random()



for i in range(250):               #Loop for the  1000 nominal cases - category 3

    for j in range (10000):

        BDDtrain[i + 500 ,j] = 2 * DataInput[j] + random.random()



for i in range(250):               #Loop for the  1000 nominal cases - category 4

    for j in range (10000):

        BDDtrain[i + 750 ,j] =  DataInput[j]

##BDD d'anomalies :

for i in range(250):               #Loop for the  1000 Anomaly cases - category 1

    for j in range (10000):

        BDDtrain[i + 1000, j] =   DataInput[j] + (random.random()*10)



for i in range(250):               #Loop for the  1000 Anomaly cases - category 2

    for j in range (10000):

        BDDtrain[i + 1250, j] =   math.cos(j/500)



for i in range(250):               #Loop for the  1000 Anomaly cases - category 3

    for j in range (10000):

        BDDtrain[i + 1500, j] =   (DataInput[j]/10) + random.random()



for i in range(250):               #Loop for the  1000 Anomaly cases - category 4

    for j in range (10000):

        if j <= 2500:

            BDDtrain[i + 1750, j] = 0

        else:

            BDDtrain[i + 1750, j] = DataInput[j] + random.random()

# Transform inputs in under-series of n_steps size

def shape_data(sequence):



    X, y = list(), list()                                         # create into new list

    seq_x = sequence[0:len(sequence)]

    X.append(seq_x)                         #append x and y during the process

    #for i in range (len(sequence)):

    seq_y = 0

    y.append(seq_y)

    seq_y =1

    y.append(seq_y)



    return array(X), array(y)   # return the new shape inputs
#Display of the 4 kinds of nominal types and the 4 types of anomaly 4.





plt.subplot(2, 2, 1)

plt.title('Nominal type 1');

plt.plot(BDDtrain[200,:])



#plt.show()

plt.subplot(2, 2, 2)

plt.title('Nominal type 2');

plt.plot(BDDtrain[400,:])

#plt.show()

plt.subplot(2, 2, 3)

plt.xlabel('Nominal type 3');

plt.plot(BDDtrain[600,:])

#plt.show()

plt.subplot(2, 2, 4)

plt.xlabel('Nominal type 4');

plt.plot(BDDtrain[800,:])

plt.show()









#display of the 4 anomalies

plt.subplot(2, 2, 1)

plt.title('Anomaly 1');

plt.plot(BDDtrain[1200,:])



#plt.show()

plt.subplot(2, 2, 2)

plt.title('Anomaly 2');

plt.plot(BDDtrain[1400,:])

#plt.show()

plt.subplot(2, 2, 3)

plt.xlabel('Anomaly 3');

plt.plot(BDDtrain[1600,:])

#plt.show()

plt.subplot(2, 2, 4)

plt.xlabel('Anomaly 4');

plt.plot(BDDtrain[1800,:])

plt.show()
# labelisation of the signals in 8 categories

for i in range (2000):

    if i < 250:

        BDDytarget[i, 0] = 1

    if  250 <=  i  < 500:

        BDDytarget[i, 1] = 1

    if  500 <=  i  < 750:

        BDDytarget[i, 2] = 1

    if  750 <=  i  < 1000:

        BDDytarget[i, 3] = 1

    if  1000 <=  i  < 1250:

        BDDytarget[i, 4] = 1

    if  1250 <=  i  < 1500:

        BDDytarget[i, 5] = 1

    if  1500 <=  i  < 1750:

        BDDytarget[i, 6] = 1

    if  1750 <=  i  < 2000:

        BDDytarget[i, 7] = 1

X, y = shape_data(DataInput)   # call of the reshape function defined before
#reshape the database to give it a NN compatible shape



X = BDDtrain.reshape((2000, 10000, n_features)) #reshape pour avoir un format d'entrée de NN

y = y.reshape(1,2)
# creation of the model and its structure

model = Sequential()



model.add(Dense(5, activation='relu',input_shape=(10000, 1)))       

model.add(Dense(3))

model.add(Dense(3))

model.add(Flatten())

model.add(Dense(8,activation = "softmax"))





model.summary()
#optimisation parameters

model.compile(optimizer='adam', loss='binary_crossentropy',metrics = ['accuracy'])    
model.fit(X, BDDytarget, epochs=10, verbose=1)      
#Application of the trained NN



listtemp =np.zeros([10000,1])





#Reshape before prediction

x_input = listtemp.reshape((1, 10000, n_features))     #reshape pour entrée en NN





#Prediction of the kind(anomaly or no)

prediction = model.predict(x_input, verbose=0)

score = prediction[0,1]

score
#UTilisation of a test database



BDDtest=np.zeros([10,10000])            # 10 for the number of signal we test

                                        

predictiontest = np.zeros([1000,2])

for i in range(10):               #loop to create one kind of signal

    for j in range (10000):

        BDDtest[i,j] =  2 * DataInput[j] + random.random()

        







BDDtest = BDDtest.reshape((10, 10000, n_features))     #reshape to get to the NN



#Test and visualisation of the results

predictiontest= model.predict(BDDtest, verbose=1)

print(predictiontest)



#Hist plot and visualisation of the tested signal

plt.bar(range(8), 100*predictiontest[1], color="#3DBA38")

plt.show()

plt.plot(BDDtest[1,:])

plt.show()
