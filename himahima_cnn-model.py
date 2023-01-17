import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
X_train = train.drop(labels = ["label"],axis = 1)
y_train = train["label"]
sns.set_palette("pastel")
ax = sns.barplot(train['label'].value_counts().index, train['label'].value_counts()/len(train))
ax.set_title("Distribution of digits in labelled data")
ax.set_ylabel("Percentage")
ax.set_xlabel("Digit")
sns.despine()

X_train.head()
X_train.isnull().sum()
X_train = X_train / 255.0
y_train = y_train / 255.0
X_train.head()

X_train = pd.get_dummies(X_train)
y_train = pd.get_dummies(y_train)
X_train.head()

from sklearn.model_selection import train_test_split
#Splitting data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=44)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
g = plt.imshow(X_train[5][:,:,0],cmap='gray')

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

# STEP 1 : Building the Model 
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# STEP 2 : Compiling the Model. .   

model.compile(optimizer ='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# Step 3 : Training . .   

history = model.fit(X_train, y_train, epochs=1)
print(model.summary())

# STEP 4 : Predicting . .   

y_pred = model.predict(X_train)

print('Prediction Shape is {}'.format(y_pred.shape))
print('Prediction items are {}'.format(y_pred[:5]))


# STEP 5 : Evaluating . .   

ModelLoss, ModelAccuracy = model.evaluate(X_train, y_train)

print('Model Loss is {}'.format(ModelLoss))
print('Model Accuracy is {}'.format(ModelAccuracy ))