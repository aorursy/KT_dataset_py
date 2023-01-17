import pandas as pd
train = pd.read_csv('../input/digit-recognizer/train.csv')
train.head(4)
test = pd.read_csv('../input/digit-recognizer/test.csv')
test.head(4)
x_train = train.iloc[:, 1:785].values
y_train = train.iloc[:, 0].values
x_test = test.iloc[:, 0:784].values
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
x_train = x_train/255.0
x_test = x_test/255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print(x_train.shape)
print(x_test.shape)
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import to_categorical
from keras.utils import np_utils
num_classes = 10
batch_size = 64
epochs = 40
input_shape = (28,28,1)

seed = 5
np.random.seed(seed)

y_train = to_categorical(y_train, num_classes)

from sklearn.model_selection import train_test_split
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = 0.2, random_state=seed)

print(x_train.shape)
print(x_validation.shape)
print(y_train.shape)
print(y_validation.shape)
model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), input_shape= input_shape, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(num_classes, activation = 'sigmoid'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()
model.fit(x_train, y_train, epochs = epochs, verbose = 1)
loss , accuracy = model.evaluate(x_validation, y_validation, verbose = 0)
print("Loss : ",loss, "Accuracy : ", accuracy)
predicted_classes = model.predict_classes(x_test)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)), "Label": predicted_classes})

submissions.to_csv("submission_digit.csv", index = False, header = True)

#from google.colab import files
#files.download('submission_digit.csv')