import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

%matplotlib inline



import tensorflow as tf

from tensorflow import keras
train_data = pd.read_csv('../input/tobigs14-mnist-competition/train_df.csv')

test_data = pd.read_csv('../input/tobigs14-mnist-competition/test_df.csv')

sample_submission = pd.read_csv("../input/tobigs14-mnist-competition/sample_submission.csv")
# 어떤 자료인지 보자

train_data.head()
X_train = train_data.drop('label',axis = 1).values

y_train = train_data['label'].values



X_test = test_data.iloc[:,1:].values
X_train.shape, y_train.shape, X_test.shape
# 하나 출력해볼까!



index = 0

image = X_train[index].reshape(28,28)

plt.imshow(image, 'gray')

plt.title('label : {}'.format(y_train[index]))

plt.show()
# 전체 784 pixel, 즉, 28*28 사이즈의 그림들임!



image_size = X_train.shape[1]

print ('image_size => {0}'.format(image_size))



# in this case all images are square

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)



print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))
# scaling



X_train = X_train.astype(np.float)

X_test = X_test.astype(np.float)

X_train /= 255

X_test /= 255



print('maximum value after scaling:', X_train.max(),

      '\nminimum value after scaling:' ,X_train.min())
# train test split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)



print(X_train.shape, y_train.shape)

print(X_val.shape, y_val.shape)
model = keras.Sequential([

    keras.layers.Dense(512, activation='relu'),

    keras.layers.Dropout(0.3),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.Dropout(0.2), 

    # dropout 대신 keras.layers.BatchNormalization() 이용 가능

    keras.layers.Dense(10, activation='softmax')

])



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(X_train, y_train, epochs=5, batch_size=100)

model.evaluate(X_val, y_val)
predictions = model.predict_classes(X_test)
sample_submission['Category'] = pd.Series(predictions)

sample_submission.head()
sample_submission.to_csv("submission.csv",index=False)
