import pandas as pd

import numpy as np



import time



import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.utils import shuffle



from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical
# Reading the Train and Test Datasets.

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
# Let's see the shape of the train and test data

print(train.shape, test.shape)
train.head()
fig=plt.figure(figsize=(14,8))

columns = 8

rows = 3

for i in range(1, rows*columns+1):

    

    digit_array = train.loc[i, "pixel0":]

    arr = np.array(digit_array)   

    image_array = np.reshape(arr, (28,28))   

    

    

    fig.add_subplot(rows, columns, i)

    plt.title("Label:"+train.loc[i,"label"].astype("str"))

    plt.imshow(image_array, cmap=plt.cm.binary)

    

plt.show()
train.isna().any().any()
test.isna().any().any()
# dividing the data into the input and output features to train make the model learn based on what to take in and what to throw out.

train_X = train.loc[:, "pixel0":"pixel783"]

train_Y = train.loc[:, "label"]



train_X = train_X / 255.0

test_X = test / 255.0



train_X = train_X.values.reshape(-1,28,28,1)

test_X = test_X.values.reshape(-1,28,28,1)

train_Y = to_categorical(train_Y, num_classes = 10)
# Let's see the shape of the train and test data

print(train_X.shape, train_Y.shape)
# Let's make some beautiful plots.

def visualize_digit(row):

    

    plt.imshow(train_X[row].reshape((28,28)),cmap=plt.cm.binary)

    plt.title("IMAGE LABEL: {}".format(train.loc[row, "label"]))



visualize_digit(25)    
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10, # randomly rotate images in the range (degrees, 0 to 180)

        fill_mode='nearest',

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally 

        height_shift_range=0.1,  # randomly shift images vertically 

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images
# PREVIEW AUGMENTED IMAGES

plt.figure(figsize=(10,5))



for i in range(50):  

    train_X_one = train_X[25,].reshape((1,28,28,1))

    train_Y_one = train_Y[25,].reshape((1,10))    

    

    plt.subplot(5, 10, i+1)

    X_train2, Y_train2 = datagen.flow(train_X_one,train_Y_one).next()

    plt.imshow(X_train2[0].reshape((28,28)),cmap=plt.cm.binary)    

    plt.axis('off')  

    

#plt.subplots_adjust(wspace=-0.1, hspace=-0.1)



plt.show()
%%time

copy_count=4 #count of new images for 1 image in train data



#create train2 - array of new images

train2=np.empty([copy_count*42000, 785])

number=0



for n in range(0,42000):

    

    #get one image from train data

    train_X_one = train_X[n,].reshape((1,28,28,1))

    train_Y_one = train_Y[n,].reshape((1,10))    

    

    if n % 1000 == 0 : print(n*copy_count)       



    for i in range(copy_count):  

        #Generate new image

        X_train2, Y_train2 = datagen.flow(train_X_one,train_Y_one).next()    

        #add label to new image

        X_train2=np.append(X_train2,train.loc[n, "label"].astype(int))        

        #add new image with label to train2

        train2[number]=X_train2.reshape(1,785)

        number=number+1       

train2.shape
train2=pd.DataFrame(train2)

train2.columns=[str("pixel"+str(x)) for x in range(0,785)]

train2=train2.rename(columns = {'pixel784':'label'}) 
train2
fig=plt.figure(figsize=(14,8))

columns = 8

rows = 3

for i in range(1, rows*columns+1):

    

    digit_array = train2.loc[i-1, "pixel0":"pixel783"]

    arr = np.array(digit_array)   

    image_array = np.reshape(arr, (28,28))   

    

    

    fig.add_subplot(rows, columns, i)

    plt.title("Label:"+train2.loc[i-1,"label"].astype("str"))

    plt.imshow(image_array, cmap=plt.cm.binary)

    

plt.show()
train_all=pd.concat([train, train2], ignore_index=True)

del train,train2

train_all['label']=train_all['label'].astype(int)
train_all
train_all=shuffle(train_all)

train_all.head()
model = RandomForestClassifier(random_state=1, n_jobs=-1)
%%time

model.fit(train_all.loc[:,'pixel0':'pixel783'], train_all.loc[:,'label'])
%%time

cv=cross_val_score(model, train_all.loc[:,'pixel0':'pixel783'], train_all.loc[:,'label'], cv=5)

print(cv, "RF  mean=", cv.mean())
test['Label'] = model.predict(test).astype(int)

### Add "ImageId" as Index+1

test['ImageId']=test.index+1

test.loc[:,['ImageId','Label']].head()
test.loc[:,['ImageId','Label']].to_csv('RF_with_DA_sub.csv', index=False)
fig=plt.figure(figsize=(14,14))

columns = 7

rows = 5

for i in range(1, rows*columns+1):

    

    digit_array = test.loc[i, "pixel0":"pixel783"]

    arr = np.array(digit_array)   

    image_array = np.reshape(arr, (28,28))   

    

    

    fig.add_subplot(rows, columns, i)

    plt.title("Predict:"+test.loc[i,"Label"].astype("str"))

    plt.imshow(image_array, cmap=plt.cm.binary)

    

plt.show()