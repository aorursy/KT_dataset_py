import numpy as np
import pandas as pd
training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
training_data.head()
x_train = training_data.drop('label', axis = 1)
y_train = pd.DataFrame(data=training_data['label'])
display(y_train.head())
display(x_train.head())
print(x_train.iloc[0].shape)
print(y_train.iloc[0].shape)
from matplotlib.pyplot import imshow
from PIL import Image
%matplotlib inline

# change the value of i to choose which image in the dataset to display
i= 1
# display the image
imshow(x_train.iloc[i].values.reshape((28, 28)))
print('This image corresponds to ', y_train.iloc[i])
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D

cnn_model = Sequential()

# convolusion layers followed by max pooling
cnn_model.add(Conv2D(128, (3,3), padding='same', input_shape=(28,28,1), data_format='channels_last', activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.2))

cnn_model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.2))

cnn_model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.2))

cnn_model.add(Conv2D(256, (3,3), padding='valid', activation='relu'))
cnn_model.add(Dropout(0.2))

# output layer
cnn_model.add(Flatten())
cnn_model.add(Dense(units=10, activation='softmax'))

# compile the model
# for a multi-class classification problem
cnn_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('The model was successfully created and compiled.')
print('Shape of Layer Outputs:')
for layer in cnn_model.layers:
    print(layer.name,': ',layer.output_shape)
from keras.utils import to_categorical

y_train_categorical = to_categorical(y_train, num_classes=10)

reshaped_x = x_train.values.reshape(x_train.shape[0],28,28,1) / 255

print(reshaped_x.shape)
print(y_train_categorical.shape)

cnn_model.fit(x=reshaped_x, y=y_train_categorical, batch_size=1000, epochs=32, verbose=1, validation_split=0.2)
test_data.head()
# reshape the test data
reshaped_test_data = test_data.values.reshape(test_data.shape[0],28,28,1) / 255

# make predictions
predictions = cnn_model.predict(reshaped_test_data)
display(predictions)
# format the predictions into numbers from 0 to 9
predictions_formatted = np.argmax(predictions, axis=1)
display(predictions_formatted)
# make a dataframe out of the predictions
submission = pd.DataFrame({'ImageId': np.arange(1,28001), 'Label': predictions_formatted})

# output a csv file
submission.to_csv('submission_4.csv', index=False)
print('Done')