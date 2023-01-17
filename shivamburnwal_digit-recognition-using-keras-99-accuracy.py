import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
train_df.head()
plt.figure(figsize=(10,8))
sns.countplot(train_df.label)
f, ax = plt.subplots(2,5) 
f.set_size_inches(10, 10)
k = 0
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(x_train_data[k].reshape(28, 28) , cmap = "gray")
        k += 1
    plt.tight_layout() 
encoder = OneHotEncoder() # used for encoding y values as this is a multiclass classification problem.
y_train_data = train_df.label.values
y_train_data = encoder.fit_transform(y_train_data.reshape(-1,1)).toarray()
del train_df['label']

x_train_data = train_df.values
x_train_data.shape, y_train_data.shape
x_test_data = test_df.values
x_test_data.shape
# normalizing our data.
scaler = MinMaxScaler(feature_range=(0,1))
x_train_data = scaler.fit_transform(x_train_data)
x_test_data = scaler.transform(x_test_data)
# Reshaping the data (i.e. images) for model from 1-D to 3-D.
x_train_data = x_train_data.reshape(-1, 28, 28, 1)
x_test_data = x_test_data.reshape(-1, 28, 28, 1)
x_train_data.shape, y_train_data.shape
# splitting training data into validation and training to train our model on.
x_train, x_validation, y_train, y_validation = train_test_split(x_train_data, y_train_data, random_state=0)
model=Sequential()
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Flatten())
model.add(Dense(units = 512 , activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(units = 10 , activation = 'softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model.summary()
rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
history = model.fit(x_train, y_train, batch_size=40, epochs=20, validation_data=(x_validation, y_validation), callbacks=[rlrp])
print("Accuracy of our model on Validation Data : " , model.evaluate(x_validation,y_validation)[1]*100 , "%")
plt.title('Accuracf training and validation data.')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
# predicting on validation data.
pred_validation = model.predict_classes(x_validation)
new_y_validation=[]
for i in y_validation:
    new_y_validation.append(np.argmax(i))

cm = confusion_matrix(new_y_validation, pred_validation)
cm = pd.DataFrame(cm , index = [i for i in range(10)] , columns = [i for i in range(10)])
plt.figure(figsize = (10,10))
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
predictions = model.predict_classes(x_test_data)
predictions[:5]
submission['Label'] = predictions
submission.to_csv("submission.csv" , index = False)
submission.head()
