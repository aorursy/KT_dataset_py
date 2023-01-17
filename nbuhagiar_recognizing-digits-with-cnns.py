import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.describe()
sns.countplot(train["label"])
train.isnull().any().describe()
test.isnull().any().describe()
train = train.sample(frac=1, random_state=0)
X_train = train.drop(labels=["label"], axis=1)
X_train = X_train / 255
Y_train = train["label"]
num_classes = len(Y_train.unique())
img_rows, img_cols = 28, 28
X_train = X_train.values.reshape(-1, img_rows, img_cols, 1)
plt.imshow(X_train[0][:, :, 0], cmap="Greys")
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss=categorical_crossentropy, optimizer="adam", metrics=["accuracy"])
sns.countplot(Y_train.tail(int(0.1*len(Y_train))))
Y_train = to_categorical(Y_train, num_classes)
model.fit(X_train, 
          Y_train, 
          epochs=50, 
          callbacks=[EarlyStopping(monitor="val_acc", patience=10)], 
          validation_split=0.1)
test = test / 255
X_test = test.values.reshape(-1, img_rows, img_cols, 1)
predictions = model.predict(X_test)
submission = pd.DataFrame(np.argmax(predictions, axis=1)).reset_index()
submission.columns = ["ImageId", "Label"]
submission["ImageId"] += 1
submission.to_csv("submission.csv", index=False)
