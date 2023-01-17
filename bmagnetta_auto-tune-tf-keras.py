import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

import math

import pickle

import os.path

from os import path

import random

import tensorflow as tf
# Load the data on local machine

#train_kaggle = pd.read_csv("train.csv")

#test_kaggle = pd.read_csv("test.csv")

# Load the data on kaggle machine

train_kaggle = pd.read_csv("../input/digit-recognizer/train.csv")

test_kaggle = pd.read_csv("../input/digit-recognizer/test.csv")



X_train_kaggle = train_kaggle.drop(labels = ["label"],axis = 1).to_numpy()

Y_train_kaggle = train_kaggle["label"]



# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train_kaggle = X_train_kaggle.reshape(-1,28,28,1)

test_kaggle = test_kaggle.to_numpy().reshape(-1,28,28,1)



print('--- check for X_train_kaggle nan:\n',np.any(np.isnan(X_train_kaggle)))

#visualize the data

plt.imshow(X_train_kaggle[1,:,:,0])

plt.show()

plt.hist(Y_train_kaggle,np.unique(Y_train_kaggle).shape[0],rwidth=0.9)

plt.show()



print('--- check for test_kaggle nan:\n',np.any(np.isnan(test_kaggle)))

#visualize the data

plt.imshow(test_kaggle[1,:,:,0])

plt.show()
(X_train_keras, Y_train_keras), (X_val_keras, Y_val_keras) = mnist.load_data()



# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train_keras = X_train_keras.reshape(-1,28,28,1)

X_val_keras = X_val_keras.reshape(-1,28,28,1)



print('--- check for X_train_keras nan:\n',np.any(np.isnan(X_train_keras)))

#visualize the data

plt.imshow(X_train_keras[1,:,:,0])

plt.show()

#sns.countplot(Y_train_keras)

plt.hist(Y_train_keras,np.unique(Y_train_keras).shape[0],rwidth=0.9)

plt.show()



print('--- check for X_val_keras nan:\n',np.any(np.isnan(X_val_keras)))

#visualize the data

plt.imshow(X_val_keras[1,:,:,0])

plt.show()

plt.hist(Y_val_keras,np.unique(Y_val_keras).shape[0],rwidth=0.9)

plt.show()
#If tf.keras has error in download (sometimes happens when running on kaggle) comment out concatenate

#X = np.concatenate((X_train_kaggle,X_train_keras,X_val_keras), axis=0)

#Y = np.concatenate((Y_train_kaggle,Y_train_keras,Y_val_keras), axis=0)

X = X_train_kaggle

Y = Y_train_kaggle



# Normalize the data

X = X / 255.0

test_kaggle = test_kaggle / 255.0



# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.05, random_state=2)



print('X.shape: ',X.shape)

print('X_train.shape = ',X_train.shape)

print('X_val.shape = ',X_val.shape)



print('Y.shape: ',Y.shape)#Note we cold have don't one-hot encoding which would yeild diff. shape

print('Y_train.shape = ',Y_train.shape)

print('Y_val.shape = ',Y_val.shape)
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(X_train)
def rand_tunning(lr,opt_name,m,r,l2,l1,num_epochs,d):

    

    if lr == None:

        lr = 10**(random.uniform(-4.,-2.))#uniformly sample     

    if opt_name == None:

        opt_name = random.choice(['sgd','adadelta','adagrad','adam','adamax','ftrl','nadam','rmsprop'])

    if m == None:

        m = 10**(random.uniform(-4.,-2))#uniformly sample      

    if r == None:

        r = 10**(random.uniform(-1.,1.))#uniformly sample      

    if l1 == None:

        l1 = 10**(random.uniform(-5.,1.))#uniformly sample       

    if l2 == None:

        l2 = 10**(random.uniform(-5.,1.))#uniformly sample       

    if d == None:

        d = random.uniform(0.4,0.6)#uniformly sample       

        

    if opt_name == 'sgd':

        opt = tf.keras.optimizers.SGD(learning_rate=lr,momentum=m)

    if opt_name == 'adadelta':

        opt = tf.keras.optimizers.Adadelta(learning_rate=lr,rho=r)

    if opt_name == 'adagrad':

        opt = tf.keras.optimizers.Adagrad(learning_rate=lr)

    if opt_name == 'adam':

        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    if opt_name == 'adamax':

        opt = tf.keras.optimizers.Adamax(learning_rate=lr)

    if opt_name == 'ftrl':

        opt = tf.keras.optimizers.Ftrl(learning_rate=lr)

    if opt_name == 'nadam':

        opt = tf.keras.optimizers.Nadam(learning_rate=lr)

    if opt_name == 'rmsprop':

        opt = tf.keras.optimizers.RMSprop(learning_rate=lr,momentum=m,rho=r)

    

    

    tunning_parameters = {'lr':lr,'opt_name':opt_name,'m':m,'r':r,'l2':l2,'l1':l1,'num_epochs':num_epochs,'d':d}

    print('tunning_parameters: ',tunning_parameters)

    

    IMG_HEIGHT=X_train.shape[1]

    IMG_WIDTH=X_train.shape[2]

    



    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH,1)),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),

        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),

        tf.keras.layers.Flatten(),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(d),

        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(d),

        tf.keras.layers.Dense(10, activation='softmax')

    ])

    

    

    model.compile(optimizer=opt,

                  loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])

    callbacks = [ tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)]

    

    fit = model.fit_generator(datagen.flow(X_train,Y_train),validation_data=(np.asarray(X_val), np.asarray(Y_val)), epochs=num_epochs, callbacks=callbacks)



    return tunning_parameters,model,fit

    
def bias_variance_visualization(parameters,historys):

    #need to comment if file doesn't exist yet, then uncomment

    if path.exists("test_tune.pkl")==True:

        with open('test_tune.pkl', 'rb') as f:

            points = pickle.load(f)

    else:

        points=[]



        

    for i,history in enumerate(historys):

        train_acc_final = history['accuracy'][-1]

        test_acc_final = history['val_accuracy'][-1]

        bv_balance = [(train_acc_final-test_acc_final),1-(train_acc_final+test_acc_final)/2,parameters[i]]

        points.append(bv_balance)

        

    with open('test_tune.pkl', 'wb') as f:

        pickle.dump(points, f)

        

    radius = [math.sqrt(points[i][0]**2+points[i][1]**2) for i in range(len(points))]

    min_rad_index = [i for i,r in enumerate(radius) if r == min(radius)][0]

    best_point = points[min_rad_index][2]

    print('\n\nbest parameter config = ', best_point)

    



    

    points=list(zip(*points))



    plt.scatter(list(points[0]),list(points[1]))

    plt.xlim([-1.5*max(np.abs(list(points[0]))),1.5*max(np.abs(list(points[0])))])

    plt.ylim([0,1])

    plt.show()

    

    return best_point







def get_child(rad_th):



    with open('test_tune.pkl', 'rb') as f:

        points = pickle.load(f)



    rads = [[math.sqrt(p[0]**2+p[1]**2),p[2]] for p in points if math.sqrt(p[0]**2+p[1]**2)<rad_th]



    rns = random.sample([i for i in range(0,len(rads))],2)



    print('\n',len(rads),' parents to choose from\n')

    

    p1 = rads[rns[0]]

    #print('parent 1: ',p1[1])

    p2 = rads[rns[1]]

    #print('parent 2: ',p2[1])



    child = {}

    for key in p1[1].keys():

        if isinstance(p1[1][key], str)==True:

            #becuase p1 and p2 are randomly set, it is ok to just take string from p1

            child[key]=p1[1][key]

        elif key in p2[1].keys():

            #both parents have value

            r=random.random()

            #random linear mapping between parent values

            child[key]=r*p1[1][key]+(1-r)*p2[1][key]

        elif isinstance(p1[1][key], int)==True:

            child[key]=int(random.uniform(0.8,1.2)*p1[1][key])

        else:

            child[key]=random.uniform(0.8,1.2)*p1[1][key]

    for key in p2[1].keys():

        if key not in p1[1].keys():

            if isinstance(p2[1][key], str)==True:

                #becuase p1 and p2 are randomly set, it is ok to just take string from p1

                child[key]=p1[1][key]

            elif isinstance(p2[1][key], int)==True:

                child[key]=int(random.uniform(0.8,1.2)*p2[1][key])

            else:

                child[key]=random.uniform(0.8,1.2)*p2[1][key]



    #add random mutation/modification

    if random.random()<0.25:

        key_mut = random.sample(p1[1].keys(),1)[0]

        child[key_mut] = None

        

    return child
for _ in range(2):

    (tunning_parameters,model,fit) = rand_tunning(lr=None,opt_name='sgd',m=None,r=0,l2=0,l1=0,num_epochs=1,d=None)



    print(tunning_parameters)

    print(fit.history)



    bias_variance_visualization([tunning_parameters],[fit.history])



for i in range(1):

    #gradually increase the number of epochs

    print('\n\n===child ',i,'===\n\n')

    child = get_child(rad_th=1)

    print('child parameters: ', child,'\n\n')

    (tunning_parameters,model,fit) = rand_tunning(lr=child['lr'],opt_name=child['opt_name'],m=child['m'],r=child['r'],l2=child['l2'],l1=child['l1'],num_epochs=1,d=child['d'])

    bias_variance_visualization([tunning_parameters],[fit.history])



    
best_point = bias_variance_visualization([tunning_parameters],[fit.history])



#best_point = {'lr': 0.0015, 'opt_name': 'nadam', 'm': 0.0026639587079509783, 'r': 0.0, 'l2': 0.0, 'l1': 0.0, 'num_epochs': 10, 'd': 0.2}

(tunning_parameters,model,fit) = rand_tunning(lr=best_point['lr'],opt_name=best_point['opt_name'],m=best_point['m'],r=best_point['r'],l2=best_point['l2'],l1=best_point['l1'],num_epochs=1,d=best_point['d'])

saved_model = tf.keras.models.load_model('best_model.h5')#change to best_model for yours

results = saved_model.predict(test_kaggle)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)