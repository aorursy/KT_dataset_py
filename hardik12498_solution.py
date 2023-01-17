import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

from matplotlib import pyplot as plt
file = np.genfromtxt("train.csv",delimiter=",",dtype="str")

train_f = []

for x in file:

    if(len(np.argwhere(x == "?"))):

        continue

    train_f.append(x)

train_f = np.array(train_f)

print(train_f.shape)
data = train_f[1:,1:-1]

labels = np.int32(train_f[1:,-1])

data[data == "Big"] = "0"

data[data == "Medium"] = "1"

data[data == "Small"] = "2"
unl = np.unique(labels)

print(unl)
print(train_f[train_f[:,2] == "?"])

data
for j,x in enumerate(train_f[0][1:-1]):

    print(x)

    un,co = np.unique(data[:,j],return_counts=True)

    for i in range(len(un)):

        print(un[i],co[i])
for i in range(data.shape[1]):

    print(train_f[0][1+i],i)

    for l in unl:

        data_l = np.float32(data[labels == l][:,i])

#         print("label",l)

#         plt.hist(data_l)

#         plt.show()

#         un,co = np.unique(data_l,return_counts=True)

#         for k in range(len(un)):

#             print(un[k],co[k])

        print(np.mean(data_l),np.median(data_l),np.std(data_l))
"Number of Quantities is related to Number of Insignificant Quantities"

"Number of Sentences is related to Number of Special Characters"

from keras.utils import to_categorical

sizes = to_categorical(data[:,2])

data = data[:,[3,4,5,7,8,9,10]]

data = np.concatenate([data,sizes],axis=1)

data.shape
from keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler



train_data = np.float32(data)

train_labels = to_categorical(labels,num_classes=6)

p = np.random.permutation(len(train_data))



train_data = train_data[p]

train_labels = train_labels[p]



means = []

stds = []



# for i in range(train_data.shape[1]):

#     mean = np.mean(train_data[:,i])

#     std =  np.std(train_data[:,i])

#     means.append(mean)

#     stds.append(std)

#     train_data[:,i] =( train_data[:,i] - mean) / std

print(train_data.shape)
train_data[:]
split = len(train_data) // 8



p = np.random.permutation(len(train_data))

train_data = train_data[p]

train_labels = train_labels[p]





from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test=train_test_split(train_data,train_labels,test_size=0.1,stratify=train_labels)



def generator(datax,datay,batch_size = 32,train=False):

    while(True):

        p = np.random.permutation(len(datax))

        

        datax = datax[p]

        datay = datay[p]

        stds = np.std(datax,axis=0)

        for i in range(0,len(datax)-batch_size,batch_size):

            batchx = list()

            batchy = list()

            for j in range(i,i+batch_size):

                dt = np.copy(datax[j])

#                 if(train):

#                     rd = (np.random.random(size=len(stds)) * 2) - 1

#                     dt = dt + (rd * (stds / 10))

                batchx.append(dt)

                batchy.append(datay[j])

            batchx = np.array(batchx)

            batchy = np.array(batchy)

            yield batchx,batchy

            



train_gen = generator(x_train,y_train,train=True)

val_gen = generator(x_test,y_test)



print(type(x_train))

next(train_gen)

for i in range(4):

    print(next(train_gen))

    

for i in range(4):

    print(next(val_gen))
import keras

from keras.layers import * 

from keras.models import Sequential

model = Sequential();

model.add(Dense(32, input_dim=10));

model.add(keras.layers.Activation("sigmoid"))

model.add(keras.layers.BatchNormalization())

model.add(Dropout(0.2));

model.add(Dense(16));

model.add(keras.layers.Activation("sigmoid"))

model.add(keras.layers.BatchNormalization())

model.add(Dropout(0.2));

model.add(Dense(6, activation='softmax'));

model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy']);



ckpt = keras.callbacks.ModelCheckpoint("model1.h5",monitor="acc",mode="max")

history = model.fit_generator(train_gen,1000,validation_data=val_gen,validation_steps=100, epochs=15000,workers=8,use_multiprocessing=16,callbacks=[ckpt]);
model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy']);



ckpt = keras.callbacks.ModelCheckpoint("model2.h5",monitor="acc",mode="max")

history = model.fit_generator(train_gen,1000,validation_data=val_gen,validation_steps=100, epochs=15000,workers=8,use_multiprocessing=16,callbacks=[ckpt]);

for i in range(10):

    print(next(train_gen))
file = np.genfromtxt("test.csv",delimiter=",",dtype="str")

test_f = []

for x in file:

#     if(len(np.argwhere(x == "?"))):

#         continue

    test_f.append(x)

test_f = np.array(test_f)

print(test_f.shape)
print(test_f[test_f[:,2] == "?"])
data = test_f[1:,1:]

# labels = np.int32(train_f[1:,-1])

data[data == "Big"] = "0"

data[data == "Medium"] = "1"

data[data == "Small"] = "2"





from keras.utils import to_categorical

sizes = to_categorical(data[:,2])

# print(sizes.shape)

data = data[:,[3,4,5,7,8,9,10]]

data = np.concatenate([data,sizes],axis=1)

data.shape



data_test = np.float32(data)





# for i in range(data_test.shape[1]):

#     mean = means[i]

#     std =  stds[i]

#     train_data[:,i] =( train_data[:,i] - mean) / std

    

out = np.argmax(model.predict(data_test),axis=1)

print(out.shape)
out
submissionf = open("norm-nonlinear6.csv","w")

submissionf.write("ID,Class\n")

for i in range(1,len(test_f)):

    submissionf.write(test_f[i][0])

    submissionf.write(",")

    submissionf.write(str(out[i-1]))

    submissionf.write("\n")

    print(test_f[i][0],out[i-1],str(out[i-1]))

submissionf.close()