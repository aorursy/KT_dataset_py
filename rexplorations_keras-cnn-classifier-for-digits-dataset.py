import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential
from keras.regularizers import l1_l2

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
traind = pd.read_csv('../input/train.csv')
testX = pd.read_csv('../input/test.csv').as_matrix()
trainY = traind.label.tolist()
xtr = traind.drop(['label'],axis=1).as_matrix()
print(xtr.shape, testX.shape)
print("Example image from the training dataset:\n")
plt.imshow(xtr[0].reshape(28,28));
plt.show()
plt.imshow(testX[0].reshape(28,28))
dim = int(np.sqrt(xtr.shape[1]))

ytr = pd.get_dummies(pd.Series(trainY)).as_matrix()
X = xtr.reshape((len(xtr), dim,dim,1)) # Alternatively, we can choose the images themselves represented as 8x8 matrices
y = ytr
n_classes = ytr.shape[1]
model = Sequential([
    Conv2D(filters = 10, kernel_size = (2,2), input_shape = (dim,dim,1), activation='relu', padding= 'same', kernel_regularizer= l1_l2(l1=0.1, l2 = 0.1)),
    MaxPool2D((2,2)),
    Dropout(0.2),
    BatchNormalization(),
    Flatten(),
    Dense(n_classes, activation="sigmoid")
])
model.summary()
opt = Adam(lr=5e-4, decay=1e-4)

model.compile(optimizer=opt, loss="categorical_crossentropy", metrics = ['accuracy'])
history = model.fit(x = X, y = y, batch_size= 24, epochs= 12, validation_split= 0.1 )
pd.DataFrame(history.history)[['loss', 'val_loss']].plot(figsize = (16,4), title = "Loss by Epoch");
pd.DataFrame(history.history)[['acc', 'val_acc']].plot(figsize = (16,4), title = "Accuracy by Epoch")
pred_tr = model.predict(X)
pred_ts = model.predict(testX.reshape(len(testX),dim, dim, 1))
pred_tr = pd.DataFrame(pred_tr, columns=[0,1,2,3,4,5,6,7,8,9])
pred_ts = pd.DataFrame(pred_ts, columns=[0,1,2,3,4,5,6,7,8,9])
def get_class(row):
    for c in pred_tr.columns:
        if row[c]==max(row):
            return c
train_pred = pred_tr.apply(get_class, axis=1)
test_pred = pred_ts.apply(get_class, axis=1)
print(train_pred.shape, test_pred.shape)
print("Confusion matrix for the training data: \n")
print(confusion_matrix(trainY, train_pred))
plt.imshow(confusion_matrix(trainY, train_pred))
plt.title("Train Predictions Confusion Matrix")
plt.show()
results = pd.concat([pd.Series(range(1,len(testX)+1,1), name = 'ImageID'), pd.Series(test_pred, name="Label")], axis=1)
for i in range(10):
    plt.figure()
    plt.imshow(testX[i].reshape(dim,dim))
    plt.title(results['Label'][i])
    plt.show()
results.head()
results.tail()
results.shape
results.to_csv("rexplorations_keras_cnn_mnist_submission_20180822_2324.csv", index = False)