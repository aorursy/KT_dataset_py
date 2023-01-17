import numpy as np # linear algebra

import math

import pandas as pd

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from tensorflow.python import keras

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,Flatten
!ls ../input/digit-recognizer/
img_rows = 28

img_cols = 28

num_classes = 26



def extract_images(flattened_image_vectors,rows,cols):

    num_images = flattened_image_vectors.shape[0]

    return np.reshape(flattened_image_vectors,(num_images,rows,cols,1))



def prep_data(raw_data):

    x = extract_images(raw_data[:,1:],img_rows,img_cols)

    out_x = x / 255

    

    y = raw_data[:,0]

    out_y = keras.utils.to_categorical(y, num_classes)

    

    return out_x,out_y
mnist_test_file = "../input/digit-recognizer/test.csv"

raw_test_data = np.loadtxt(mnist_test_file,skiprows=1,delimiter=',')
# raw_test_data.shape

x_test = extract_images(raw_test_data,img_rows,img_cols)
mnist_train_file = "../input/digit-recognizer/train.csv"

raw_data = np.loadtxt(mnist_train_file,skiprows=1,delimiter=',')

x,y = prep_data(raw_data)
mnist_model = Sequential()

mnist_model.add(Conv2D(32,kernel_size=5,input_shape=(img_rows,img_cols,1),activation='relu'))

mnist_model.add(Conv2D(32,kernel_size=5,activation='relu'))

mnist_model.add(Flatten())

mnist_model.add(Dense(100,activation='relu'))

mnist_model.add(Dense(num_classes,activation='softmax'))

mnist_model.summary()



mnist_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

mnist_model.fit(x,y,batch_size=100,epochs=30,validation_split=0.2)
print("\t\t|\t".join(mnist_model.metrics_names))

print("-"*35)

print("\t|\t".join(map(lambda x:str(x),mnist_model.test_on_batch(x,y))))
predictions =  mnist_model.predict_classes(x_test)

train_predictions = mnist_model.predict_classes(x)



fig=plt.figure(figsize=(12,12))

for i in range(8):

    fig.add_subplot(4,4,i+1)

    plt.imshow(x_test[i].reshape(img_rows,img_cols),cmap='gray')

    plt.title(f'model classified as {predictions[i]}')

plt.show()
predicted_classes = mnist_model.predict_classes(x)

invalid_classifications = predicted_classes!=np.argmax(y,axis=1)

wrongly_classified_image_indices = np.argwhere(invalid_classifications==True).flatten()
fig=plt.figure(figsize=(12,7))

wrongly_classified_image_indices = np.random.choice(np.argwhere(invalid_classifications==True).flatten(),size=8)

invalid_count = len(wrongly_classified_image_indices)

for index,image_index in enumerate(wrongly_classified_image_indices):

    fig.add_subplot(math.ceil(invalid_count/4),4,index+1)

    plt.imshow(x[image_index].reshape(img_rows,img_cols),cmap='gray')

    plt.title(f'marked  {predicted_classes[image_index]} | actually  {np.argmax(y,axis=1)[image_index]}')

plt.show()

    
np.savetxt('submit1.csv',[(i+1,j) for i,j in enumerate(predictions)],fmt="%d,%d",delimiter=',',header='ImageId,Label',comments='')
!head submit1.csv