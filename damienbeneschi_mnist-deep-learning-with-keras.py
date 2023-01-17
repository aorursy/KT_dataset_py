import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
#Loading dataset
train = pd.read_csv('../input/train.csv')

#Reshape data as images 28x28 pixels
predictors = train.drop(columns=['label']).values.reshape(42000, 28,28,1)
#Scaling improves the optimization of the neural network
predictors = predictors / 255

#Get target variable as dummy variables
target = to_categorical(train['label'])

print("Shapes of predictors and target arrays:", predictors.shape, target.shape)
#Building of the neural network
cnn = Sequential()
#Input layer
cnn.add(Conv2D(100, kernel_size=(2,2), activation='relu', input_shape=(28,28,1)))
#Additional convolutional layers
cnn.add(Conv2D(100, kernel_size=(2,2), activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Conv2D(100, kernel_size=(2,2), activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Conv2D(100, kernel_size=(2,2), activation='relu'))
cnn.add(Dropout(0.5))
#Flattening of the convolutionnal layer
cnn.add(Flatten())
#Dense layer between the flattening and output layers
cnn.add(Dense(200, activation='relu'))
#Output layer for 10 digits with probability values
cnn.add(Dense(10, activation='softmax'))

#Compile the model
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#Training of the model
early_stopping_monitor = EarlyStopping(patience=3)
cnn.fit(predictors, target, validation_split=0.3, epochs=50)#, callbacks=[early_stopping_monitor])
#Prepare the submission dataset
test = pd.read_csv('../input/test.csv')
submission = test.values.reshape(len(test), 28, 28,1) / 255
print("Submission dataset shape: ", submission.shape)

#Make predictions on the test set
predictions = cnn.predict(submission)  #Contains probabilities for each 10 digits
print("Predictions shape (contains pseudo-probabilities for each digit type) :", predictions.shape)

#Creating the submission dataframe
submission_df = pd.DataFrame(predictions)
submission_df['ImageId'] = pd.Series(range(1, 28001))
#Take the corresponding column label for the max value (probability) of the row
submission_df['label'] = submission_df.drop(columns=['ImageId']).idxmax(1)
print("20 first predictions on the submission digits: \n", submission_df[['label']].head(20))

#Print the 10 first images of the submission dataset
images_subset = test.iloc[:20, :]
for i, row in images_subset.iterrows():
    plt.subplot(2, 10, i+1)
    pixels = row.values.reshape((28,28))
    plt.imshow(pixels, cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.title('20 first images from the submission dataset')
plt.show()
#Save the submission as csv
submission_df.loc[:, ['ImageId', 'label']].to_csv('submission_cnn_DB.csv', index=False)