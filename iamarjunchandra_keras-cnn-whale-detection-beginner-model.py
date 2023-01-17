import numpy as np

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

from IPython.display import HTML

import os

print(os.listdir("../input"))



%matplotlib inline



df=pd.read_csv('../input/train.csv')

df.head()
df.count()
df['Path']=df['Image'].map(lambda x:'../input/train/{}'.format(x))

df.head()
df['Id'].nunique()
df['Id'].value_counts().head(20)
random_whale=np.random.choice(df['Path'],2)

for whale in random_whale:

    image=Image.open(whale)

    plt.imshow(image)

    plt.show()
from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input



def add_img(dataset,shape,img_size):

    

    x_train = np.zeros((shape, img_size[0], img_size[1], img_size[2]))

    count = 0

    

    for fig in dataset.itertuples():

        

        #load train data images into images of specified size

        img = image.load_img(fig.Path, target_size=img_size)

        x = image.img_to_array(img)

        x = preprocess_input(x)

        x_train[count] = x

        count += 1

    

    return x_train
from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical

def label(y):

    y_train=np.array(y)

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(y_train)

    y_train = to_categorical(y_train, num_classes = 5005)

    return y_train,label_encoder
x_train=add_img(df,df.shape[0],(100,100,3))

y_train,encoder=label(df['Id'])

x_train/=255 #Normalizing the data
y_train.shape
# Importing the Keras packages

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator
classifier = Sequential()
classifier.add(Convolution2D(16, 5, 5, input_shape = (100,100, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(16, 5, 5, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.25))
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.25))
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.25))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 240, activation = 'relu'))

classifier.add(BatchNormalization())

classifier.add(Dense(output_dim = y_train.shape[1], activation = 'sigmoid'))
from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau



# Define the optimizer

adam_optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)



# Set a learning rate annealer

learning_rate = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
classifier.compile(optimizer = adam_optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.summary()
whale_detector = classifier.fit(x_train, y_train, epochs=60, batch_size=1000, verbose=10, callbacks=[learning_rate])
test = os.listdir("../input/test/")

test_df = pd.DataFrame(test, columns=['Image'])

test_df['Path']=test_df['Image'].map(lambda x:'../input/test/{}'.format(x))

x_test=add_img(test_df,test_df.shape[0],(100,100,3))

x_test/255

pred=classifier.predict(np.array(x_test),verbose=1)#Since numpy array is faster than df
# Plot the loss curve for training

plt.plot(whale_detector.history['loss'], color='r', label="Train Loss")

plt.title("Train Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
# Plot the accuracy curve for training

plt.plot(whale_detector.history['acc'], color='g', label="Train Accuracy")

plt.title("Train Accuracy")

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
test_df['Id']=''

for index,prediction in enumerate(pred):

    test_df.loc[index, 'Id'] = ' '.join(encoder.inverse_transform(prediction.argsort()[-5:][::-1]))

test_df.drop(['Path'],axis=1,inplace=True)

test_df.to_csv('submission.csv', index=False)

test_df.head()