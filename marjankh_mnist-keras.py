import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.describe()
train['label'].unique()
train['label'].count()
X_train = train.drop(labels = ["label"], axis = 1)  

Y_train = train["label"]                            

X_train.head()
X_test = test.values    

X_test.shape
import seaborn as sns

sns.countplot(Y_train)
# Check for null and missing data



#X_train.isnull().any()

X_train.isnull().any().describe()
Y_train.isnull().any()
# Normalization 



X_train = X_train/255.0

X_test = X_test/255.0



X_train.shape, X_train.shape
X_train.values
# Reshape image from a 1D array with 784 elements (784 features corresponding to the 784 pixels present in the image)

# to a 3D list having dimensions of 28x28x1. 



# Reshape image in 3 dimensions (height = 28px, width = 28px, canal = 1) 

X_train = X_train.values.reshape(-1, 28, 28, 1)

X_test = X_test.reshape([-1, 28, 28, 1])
X_train.shape
# Label Encoding



from keras.utils.np_utils import to_categorical



# Convolutional Neural Network ---> Converting output into vector containing categorical values of the output



Y_train = to_categorical(Y_train, num_classes=10)
# Spliting training data & validation set



random_seed = 2



from sklearn.model_selection import train_test_split



X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, random_state=random_seed, test_size = 0.3)
from keras import Sequential

from keras.layers import Dense, MaxPooling2D, Dropout, Flatten, Conv2D
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)))

model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))

model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'] )
model.fit(batch_size=200, epochs=12, x=X_train, y=Y_train, validation_data=[X_val, Y_val])
prediction = model.predict(X_test)

#print(prediction)
import numpy as np 



label = np.argmax(prediction, axis=1)
test_id = np.reshape(range(1, len(prediction) + 1), label.shape)

print(test_id)
pred = pd.DataFrame({'ImageId': test_id, 'Label': label})



pred.head()
my_submission = pd.DataFrame({'ImageId': test_id, 'Label': label})



my_submission.to_csv('submission.csv', index=False)
# Convert DataFrame to a csv file 



filename = 'MNIST.csv'



my_submission.to_csv(filename,index=False)



print('Saved file: ' + filename)