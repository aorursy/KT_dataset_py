import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
from IPython.display import clear_output
CATEGORIES = os.listdir("/kaggle/input/plant-seedlings-classification/train")
CATEGORIES
X=[]
Y=[]
IMG_S1,IMG_S2 = 140,140        #size of the image we will use...feel free to change it and observe model behaviour
def createTrainingData():
    """
    To create training data from the dataset:
        Steps performed by this function:
            a. get the COLORED image
            b. resize image to required size
            c. convert the colours(this step does not affect the model)
                >> cv2 reads image in the format BGR. we change the color format so that we can view the image with original colors.
                >> Having different format won't change the performance of the model.
                >> You can assume that we still have all the data just kinda shuffled if we dont convert it.
            d. add image to the training set and its corresponding class
            e. flip image upside down and mirror it -> add this image also to the training set with its label
                >> This step is used to increase the amount of data we have,cosider a seedling flipped will still be a seedling.
    
    """
    a=0
    pathh = "/kaggle/input/plant-seedlings-classification/"
    types = ["train"]
    for typ in types:
        datadir = os.path.join(pathh,typ)
        for ele in CATEGORIES:
            PATH = os.path.join(datadir,ele)
            class_num = CATEGORIES.index(ele)
            for img in os.listdir(PATH):
                image = os.path.join(PATH, img)
                image = cv2.imread(image, cv2.IMREAD_ANYCOLOR)
                image = cv2.resize(image , (IMG_S1, IMG_S2))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                X.append(image)
                X.append(cv2.flip(image,-1))
                Y.append(class_num)
                Y.append(class_num)
                a+=1
        clear_output()
        print(a)
        
createTrainingData() #create the set
X = np.array(X)/255.
Y = np.array(Y)
X.shape
def one_hottie(labels,C):
    """
    One hot Encoding is used in multi-class classification problems to encode every label as a vector of binary values
        eg. if there are 3 class as 0,1,2
            one hot vector for class 0 could be : [1,0,0]
                           then class 1: [0,1,0]
                           and class 2: [0,0,1]
    We need this encoding in out labels for the model learns to predict in a similar way.
    
    Without it,if only integer values are used in labels,it could affect model in different ways,
        such as predicting a class that does not exist.
        
    """
    One_hot_matrix = tf.one_hot(labels,C)
    return tf.keras.backend.eval(One_hot_matrix)
Y = one_hottie(Y, 12)
print ("Y shape: " + str(Y.shape))

plt.imshow(X[7])
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 8)   #split the data in
del X
del Y
# Implements the forward propagation for the model:
# CONV2D -> Leaky RELU -> MAXPOOL -> CONV2D -> Leaky RELU -> MAXPOOL -> CONV2D -> Leaky RELU -> MAXPOOL -> CONV2D -> Leaky RELU -> MAXPOOL-> 
# -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
#SOME DROPout layers for regularization
# Batch normalization helps compute faster
# regularizers for regularization
# Feel free to try different values

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 1, activation=None,kernel_regularizer=tf.keras.regularizers.l2(0.1), input_shape=(140,140,3)),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.LeakyReLU(0.1),
    tf.keras.layers.MaxPool2D(strides=2),
    
    tf.keras.layers.Conv2D(128, 3, activation=None,padding="same",kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.LeakyReLU(0.1),
    tf.keras.layers.MaxPool2D(strides=2),
    
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Conv2D(256, 5, activation=None,kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.LeakyReLU(0.1),
    tf.keras.layers.MaxPool2D(strides=2),
    
    tf.keras.layers.Conv2D(64, 5, activation=None,padding="same",kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.LeakyReLU(0.1),
    tf.keras.layers.MaxPool2D(strides=2),
    
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100,kernel_regularizer=tf.keras.regularizers.l2(0.01), activation=None),
    tf.keras.layers.BatchNormalization(axis=1),
    tf.keras.layers.ReLU(),
    
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(50,kernel_regularizer=tf.keras.regularizers.l2(0.01), activation=None),
    tf.keras.layers.BatchNormalization(axis=1),
    tf.keras.layers.ReLU(),
    
    tf.keras.layers.Dense(12, kernel_regularizer=tf.keras.regularizers.l2(0.01) ,activation='softmax')
])

initial_learning_rate = 0.001 #initial rate
# Rate decay with exponential decay
# new rate = initial_learning_rate * decay_rate ^ (step / decay_steps)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=800,
    decay_rate=0.5,
    staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
Y_train.shape
result = model.fit(x=X_train,y=Y_train,batch_size=64,epochs=120,verbose=1,shuffle=False,initial_epoch=0,
                   validation_split=0.2)
result.history.keys()
plt.plot(result.history['accuracy'], label='train')
plt.plot(result.history['val_accuracy'], label='valid')
plt.legend(loc='upper left')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
plt.plot(result.history['loss'], label='train')
plt.plot(result.history['val_loss'], label='test')
plt.legend(loc='upper right')
plt.title('Model Cost')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()
valid = model.evaluate(X_test,Y_test,verbose=2)
Records = []
Records.append(valid)
Records
X=[]
file = []
def createTestData():
    a=0
    pathh = "/kaggle/input/plant-seedlings-classification/"
    types = ["test"]
    for typ in types:
        PATH = os.path.join(pathh,typ)
        for img in os.listdir(PATH):
            file.append(img)
            image = os.path.join(PATH, img)
            image = cv2.imread(image, cv2.IMREAD_ANYCOLOR)
            image = cv2.resize(image , (IMG_S1, IMG_S2))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            X.append(image)
            a+=1
    print(a)
createTestData()
X = np.array(X)/255. #normalize the data 
file
X.shape
species = model.predict_classes(X)
print(species.shape)
print(species)
ans = pd.DataFrame(file,columns = ["file"])
ans = ans.join(pd.DataFrame(species,columns=["species"]))
ans["species"] = ans["species"].apply(lambda x: CATEGORIES[int(x)])
ans.head(20)
ans.to_csv("answers.csv",index=False)
model.save("saved_model")