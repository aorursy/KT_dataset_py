import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv("../input/digit-recognizer/train.csv")
df_test = pd.read_csv("../input/digit-recognizer/test.csv")
df_train.head()
y = df_train["label"]
df_train.drop(["label"], axis=1, inplace=True)
df_train.head()
df_train = df_train.values.reshape(-1, 28, 28, 1)
df_train.shape
df_train = df_train/255.0
sns.barplot(x = y.unique(), y = y.value_counts())
plt.ylabel('Digit Frequency')
plt.xlabel('Digit')
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df_train, y, test_size=0.25, random_state=42)
from tensorflow import keras   

m = keras.Sequential()
m.add(keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
m.add(keras.layers.BatchNormalization())

m.add(keras.layers.Conv2D(32, (3, 3), activation="relu", padding='same', kernel_initializer='he_uniform'))
m.add(keras.layers.BatchNormalization())
m.add(keras.layers.MaxPool2D(2, 2))
m.add(keras.layers.Dropout(0.20))

m.add(keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same', kernel_initializer='he_uniform'))
m.add(keras.layers.BatchNormalization())

m.add(keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same', kernel_initializer='he_uniform'))
m.add(keras.layers.BatchNormalization())
m.add(keras.layers.MaxPool2D(2, 2))
m.add(keras.layers.Dropout(0.2))

m.add(keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same', kernel_initializer='he_uniform'))
m.add(keras.layers.BatchNormalization())

m.add(keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same', kernel_initializer='he_uniform'))
m.add(keras.layers.BatchNormalization())
m.add(keras.layers.MaxPool2D(2, 2))
m.add(keras.layers.Dropout(0.2))

m.add(keras.layers.Flatten())
m.add(keras.layers.Dropout(0.2))
m.add(keras.layers.Dense(128, activation="relu"))
m.add(keras.layers.Dense(10, activation="softmax"))
m.compile(optimizer=keras.optimizers.RMSprop(), loss="sparse_categorical_crossentropy", metrics=["acc"])
lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.1, 
                                            min_lr=0.003)
history = m.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=20, validation_batch_size=20, callbacks=[lr_reduce])
import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
df_test = df_test.values.reshape(-1,28,28,1)
df_test = df_test/255.0
pred = np.argmax(m.predict(df_test), axis = 1)
ans = pd.DataFrame({"ImageId" : range(1, 28001), "Label":pred})
ans.head()
ans.to_csv("official_submission.csv", index=False)
from matplotlib import pyplot as plt

# Change "cnt" to any value between 0 and 28000 to see the prediction and the correct label
cnt = 512
print("Image label is:| ", pred[cnt])
plt.imshow(df_test[cnt][:,:,0])