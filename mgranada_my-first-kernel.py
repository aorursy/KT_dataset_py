# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import time

from keras.utils.np_utils import to_categorical

from keras.models import  Sequential

from keras.layers.core import  Lambda , Dense, Flatten, Dropout

from keras.optimizers import RMSprop

from keras.layers import Conv2D  

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from keras.layers.normalization import BatchNormalization







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

print(train.head())

test= pd.read_csv("../input/test.csv") 
labels=train.label

total=[]

for i in range(10):

    curr_=sum(sum([labels == i]))

    total+=[curr_]

    #print(i,": ",curr_)

print("The diffence between the highest number of examples and the least is: ", max(total)-min(total))
print("The minimum is %.1f%% of the maximum"%((min(total))/max(total)*100))

total=np.array(total)

total=total/max(total)*100

fig=plt.figure(figsize=[12,6.5])

rec=plt.bar(range(10),total, color="SkyBlue",label="Percentages")

plt.ylabel('Number of examples')

plt.xlabel('Category')

plt.title('Number of example by number')

plt.legend()

plt.xticks(range(10));

for rect in rec:

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width()*0.5, 1.01*height,

        '%.2f'%(height), ha='center', va='bottom')

x_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

x_test = test.values.astype('float32')

print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28)

print(x_train.shape)

print(y_train.shape)





y_train= to_categorical(y_train) # We need to pass the value of the label to a vector of probability of beeing that value

print(y_train.shape)
# fix random seed for reproducibility

seed = 1234

np.random.seed(seed)

epochs=20

val_split=0.3







history=[]



model_1=Sequential() 

model_1.add(Flatten(input_shape=(28,28)))

model_1.add(Dense(10, activation='softmax'))



model_1.compile(optimizer=RMSprop(lr=0.001),

    loss='categorical_crossentropy',

    metrics=['accuracy'])

time0 = time.time()

history += [model_1.fit(x=x_train,y=y_train, epochs=epochs, verbose=0,validation_split=val_split)]

print("Done in %.3fs"%(time.time()-time0))
def plot_his(hist):

    i=1

    for history in hist:

        fig=plt.figure(figsize=[12,6.5])

        plt.plot(history.epoch,np.array(history.history['val_acc']), label='validation')

        plt.plot(history.epoch,np.array(history.history['acc']), label='tranning')

        plt.legend(["validation accuracy 1","training accuracy 1"])

        plt.title("test plot %d"%(i))

        plt.xlabel("epoch")

        plt.ylabel("accuracy %")

        plt.show()

        i+=1

    

plot_his(history)
model_2=Sequential() 

model_2.add(Flatten(input_shape=(28,28)))

model_2.add(Dropout(0.3))

model_2.add(Dense(10, activation='softmax'))



model_2.compile(optimizer=RMSprop(lr=0.001),

    loss='categorical_crossentropy',

    metrics=['accuracy'])

time0 = time.time()

history += [model_2.fit(x=x_train,y=y_train, epochs=epochs, verbose=0,validation_split=val_split)]

print("Done in %.3fs"%(time.time()-time0))


def disp_data(hist):

    print("Valid\tAVG %\t STD_TR \t STD")

    i=1;

    for history in hist:

        val1=np.array(history.history['val_acc'][-25:])

        val_tra=np.array(history.history['acc'][-25:])

        print("Hist%d:\t%.2f\t%f\t%f"%(i,np.mean(val1)*100,np.std(val_tra),np.std(val1)))

        i+=1

disp_data(history)
#notice that we have to import this new kind of layer

x_train = x_train.reshape(x_train.shape[0], 28, 28,1)

model_3=Sequential() 

model_3.add(Conv2D(filters = 6,strides=2, kernel_size=(3,3),input_shape=(28,28,1)))

model_3.add(Dropout(0.3)) 

#notice how the Conv2D layer need an extra input value that other types of layer don't 

model_3.add(Flatten())

model_3.add(Dense(10, activation='softmax'))



model_3.compile(optimizer=RMSprop(lr=0.001),

    loss='categorical_crossentropy',

    metrics=['accuracy'])

time0 = time.time()

history += [model_3.fit(x=x_train,y=y_train, epochs=epochs, verbose=0,validation_split=val_split)]

print("Done in %.3fs"%(time.time()-time0))

x_train = x_train.reshape(x_train.shape[0], 28, 28)

model_4=Sequential() 

model_4.add(Flatten(input_shape=(28,28)))

model_4.add(Dense(256, activation = "relu"))

model_4.add(Dropout(0.3))

model_4.add(Dense(10, activation='softmax'))



model_4.compile(optimizer=RMSprop(lr=0.001),

    loss='categorical_crossentropy',

    metrics=['accuracy'])

time0 = time.time()



history += [model_4.fit(x=x_train,y=y_train, epochs=epochs, verbose=0,validation_split=val_split)]

print("Done in %.3fs"%(time.time()-time0))
x_train = x_train.reshape(x_train.shape[0], 28, 28,1)

model_5=Sequential() 

model_5.add(Conv2D(filters = 6,strides=2, kernel_size=(3,3),input_shape=(28,28,1)))

model_5.add(Dropout(0.3))

model_5.add(Flatten())

model_5.add(Dense( 128, activation = "relu"))

model_5.add(Dropout(0.3))

model_5.add(Dense(10, activation='softmax'))



model_5.compile(optimizer=RMSprop(lr=0.001),

    loss='categorical_crossentropy',

    metrics=['accuracy'])

time0 = time.time()



history += [model_5.fit(x=x_train,y=y_train, epochs=epochs, verbose=0,validation_split=val_split)]

print("Done in %.3fs"%(time.time()-time0))


def make_final_model():

    model=Sequential() 

    model.add(Conv2D(filters = 12, strides=2, kernel_size=(3,3),input_shape=(28,28,1)))

    model.add(Dropout(0.3))

    model.add(BatchNormalization(axis=1))

    model.add(Conv2D(filters = 9, strides=2, kernel_size=(3,3),input_shape=(28,28,1)))

    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(128, activation = "relu"))

    model.add(Dropout(0.3))

    model.add(Dense(10, activation='softmax'))



    model.compile(optimizer=RMSprop(lr=0.001),

        loss='categorical_crossentropy',

        metrics=['accuracy'])

    return model



epochs=10



model_6 = make_final_model()

time0 = time.time()

history += [model_6.fit(x=x_train,y=y_train, epochs=epochs, verbose=0,validation_split=val_split)]

print("Done in %.3fs"%(time.time()-time0))



#plot_his(history)

#disp_data(history)
model_7 = make_final_model()



x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_split, random_state=42)



gen_t = ImageDataGenerator(rotation_range=9, width_shift_range=2, height_shift_range=2,zoom_range=0.01)

train_gen=gen_t.flow(x_train,y_train, batch_size=128)



gen_v = ImageDataGenerator()

val_gen=gen_v.flow(x_val,y_val)



time0 = time.time()

history += [model_7.fit_generator(train_gen, validation_data=val_gen, steps_per_epoch=train_gen.n/4, validation_steps=val_gen.n, epochs=2)]

print("Done in %.3fs"%(time.time()-time0))

# 

index=[]

for ii in range(len(y_val)):

    for i in range(len(y_val[ii])):

        if y_val[ii][i]==1:

            index+=[i]

            continue

            

def calcF1S(model):

    predictions=[0]*30

    for ii in range(len(y_val)):

        x_vall=x_val[ii].reshape(1,28,28,1)

        pred = model.predict_classes(x_vall)

        if pred[0] == index[ii]:

            predictions[index[ii]*3]+=1 # true positive

        else:

            predictions[pred[0]*3+1]+=1      # false positive

            predictions[index[ii]*3+2]+=1 # false negative

    # this next for loop will prevent a division by zero when calculating recall 

    # or precision

    # if no value of a class is find this code will havoid a divisioon by zero

    # while stile showing F1Score as being zero.

    for i in range(10):

        if predictions[i*3] == 0:

            if predictions[i*3+1] == 0:

                predictions[i*3+1] = 1

            if predictions[i*3+2] == 0:

                predictions[i*3+2] = 1

    pre=[0]*10

    rec=[0]*10

    F1S=[0]*10

    for i in range(10):

        #precision

        pre[i]=predictions[i*3]/(predictions[i*3]+predictions[i*3+1])

        #recall

        rec[i]=predictions[i*3]/(predictions[i*3]+predictions[i*3+2])

        #F1Score

        if(pre[i] == 0 and rec[i] == 0):

            F1S[i] = 0 ;

        else:

            F1S[i]=2*(pre[i]*rec[i]/(pre[i]+rec[i]))*100

    return F1S

       

F1S3=calcF1S(model_3)

F1S5=calcF1S(model_5)

F1S6=calcF1S(model_6)

F1S7=calcF1S(model_7)

    

ind = np.arange(len(F1S6))  # the x locations for the groups

width = 0.3

fig=plt.figure(figsize=[14,6.5])

rec1=plt.bar(ind - width/2-0.15, F1S6, width, 

                color='SkyBlue', label='Model 6')

rec2=plt.bar(ind + width/2+0.15, F1S7, width,

                color='IndianRed', label='Model 7')

rec3=plt.bar(ind, total, width,

                color='mediumseagreen', label='total')

plt.ylabel('Accuracy % and % of Total # exemples')

plt.xlabel('Labeled Number')

plt.title('Number of example by number')

plt.xticks(range(10));

plt.legend(loc=3)

for rect in rec1:

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width()*0.5, 1.01*height,

        '%.1f'%(height), ha='center', va='bottom')

for rect in rec2:

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width()*0.5, 1.01*height,

        '%.1f'%(height), ha='center', va='bottom')

for rect in rec3:

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width()*0.5, 1.01*height,

        '%.1f'%(height), ha='center', va='bottom')

    

disp_data(history)

#plot_his(history)

#disp_data(history)
data=pd.DataFrame({"Model 3":F1S3,"Model 5":F1S5,"Model 6":F1S6,"Model 7":F1S7,"Total": total})

print("Correlation between Model 3 preditions and #of examples: %.2f"%(data["Model 3"].corr(data["Total"])))

print("Correlation between Model 5 preditions and #of examples: %.2f"%(data["Model 5"].corr(data["Total"])))

print("Correlation between Model 6 preditions and #of examples: %.2f"%(data["Model 6"].corr(data["Total"])))

print("Correlation between Model 7 preditions and #of examples: %.2f"%(data["Model 7"].corr(data["Total"])))

x_test=x_test.reshape(x_test.shape[0],28,28,1)

final_predictions = model_7.predict_classes(x_test)

submissions=pd.DataFrame({"ImageId": list(range(1,len(final_predictions)+1)),"Label": final_predictions})

submissions.to_csv("Submission.csv", index=False, header=True)

print("Done!")
