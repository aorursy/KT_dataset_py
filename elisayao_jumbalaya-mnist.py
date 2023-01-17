# check input files



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# load fundamental packages



import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
# load keras package



from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import SGD, Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler, EarlyStopping
train_file = "../input/train.csv"

test_file = "../input/test.csv"

output_file = "submission.csv"
# load training data

                      #path    #skip labels  #specify datatype

raw_data = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')
# set seed for random state



np.random.seed(2017)
# train-test split

#x_train #x_test #y_train #y_test                #predictors    #response(labels)

x_train, x_val, y_train, y_val = train_test_split(raw_data[:,1:],raw_data[:,0],test_size=0.3)
print("Original dataset:")

print("training x: ", "shape ", x_train.shape, " type ", type(x_train))

print("training y: ", "shape ", y_train.shape, " type ", type(y_train))

print("validation x: ", "shape ", x_val.shape, " type ", type(x_val))

print("validation y: ", "shape ", y_val.shape, " type ", type(y_val))
# scale the data

scale_pixel = 255.0



# reshape the data

x_train = x_train.astype('float32')/255.0

x_val = x_val.astype('float32')/255.0



n_train = x_train.shape[0] # number of training observations

n_val = x_val.shape[0] # number of test observations



#x_train = x_train.reshape(n_train, 28, 28, 1)

#x_val = x_val.reshape(n_test, 28, 28, 1)



# convert integer labels to dummy variables

y_train = np_utils.to_categorical(y_train)

y_val = np_utils.to_categorical(y_val)
# dimensions of training and testing set after normalization

# we don't reshape here because for the first round we run base model



print("After normalization:")

print("training x: ", "shape ", x_train.shape, " type ", type(x_train))

print("training y: ", "shape ", y_train.shape, " type ", type(y_train))

print("validation x: ", "shape ", x_val.shape, " type ", type(x_val))

print("validation y: ", "shape ", y_val.shape, " type ", type(y_val))
# model parameters



train_size = 20000



# random sampling from training set

idx = np.random.choice(np.arange(n_train), train_size, replace=False) # 29400 choose 20000

x_sample = x_train[idx]

y_sample = y_train[idx]



# hyperparameters

learning_rate = 0.5

batch_size = 300

epoch_num = 20
# base model architecture



model = Sequential()

model.add(Dense(300, activation='relu', input_dim=(784)))  #784=28*28 pixels as input neurons

model.add(Dense(100, activation='relu'))

model.add(Dense(10, activation='softmax'))
# compile the model



model.compile(optimizer=SGD(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_sample, y_sample, validation_split=0.3, batch_size=batch_size, epochs=epoch_num)
# initiate a list of sizes



list_train_size = [2e2, 4e2, 6e2, 8e2, 1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3,

                   1e4, 2e4]



list_hist = [] # a list to store history

list_loss = [] 

list_accuracy = [] 

list_FOM = [] # a list to store FOM (Figure of Merit)



for sz in list_train_size:

    # sample [sz] number of instances from training set

    idx_sp = np.random.choice(np.arange(n_train), int(sz), replace=False) # 29400 choose [sz]

    x_sp = x_train[idx_sp]

    y_sp = y_train[idx_sp]

    

    # feed the data into the base model

    hist = model.fit(x_sp, y_sp, validation_split=0.3, batch_size=batch_size, epochs=epoch_num, verbose=0)

    score = model.evaluate(x_val, y_val)

    loss = score[0]

    p1 = len(x_sp)/len(x_train)

    p2 = score[1] # accuracy

    FOM = p1/2 + (1-p2)

    

    list_hist.append(hist)

    list_loss.append(loss)

    list_accuracy.append(p2)

    list_FOM.append(FOM)

    
# plot the result of changing training size



plt.plot(list_train_size, list_loss, label='Validation Loss')

plt.plot(list_train_size, list_accuracy, label='Validation Accuracy')

plt.plot(list_train_size, list_FOM, label='FOM')

plt.legend()