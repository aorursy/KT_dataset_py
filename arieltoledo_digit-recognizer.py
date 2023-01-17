# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



training_images = pd.read_csv("../input/digit-recognizer/train.csv")

test_images = pd.read_csv("../input/digit-recognizer/test.csv")



training_images.reset_index()



def pixel_mat(row):

    # we're working with train_df so we want to drop the label column

    vec = training_images.drop('label', axis=1).iloc[row].values

    # numpy provides the reshape() function to reorganize arrays into specified shapes

    pixel_mat = vec.reshape(28,28)

    return pixel_mat



random_num = np.random.randint(0,42000)

img = pixel_mat(random_num) 



img = img[~np.all(img == 0, axis=1)]

img = img[:,~np.all(img == 0, axis=0)]

print(img)



plt.imshow(img, cmap='gray')

plt.show()





train_data = (training_images.iloc[:,1:].values).astype('float32') # all pixel values

train_labels = training_images.iloc[:,0].values.astype('int32') # only labels i.e targets digits



test_data = test_images.values.astype('float32')



train_data = np.array(train_data.reshape(train_data.shape[0], 28, 28, 1))

test_data = np.array(test_data.reshape(test_data.shape[0], 28, 28, 1))



model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),

  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(10, activation='softmax')

])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(train_data, train_labels, epochs=10, batch_size=50)

#test_loss = model.evaluate(test_images, test_labels)



predictions = model.predict_classes(test_data)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("submission.csv", index=False, header=True)