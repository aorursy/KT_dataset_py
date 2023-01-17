import keras
import numpy as np
import pandas as pd
from PIL import Image as img
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
keras.backend.image_data_format()
keras.backend.set_image_data_format("channels_first")
keras.backend.image_data_format()
img.open("../input/train/train/dog.10001.jpg").resize((64, 64)).convert("L")
img.open("../input/train/train/cat.10001.jpg").resize((64, 64)).convert("L")
dog_train_list = glob.glob("../input/train/train/dog.*.jpg")
cat_train_list = glob.glob("../input/train/train/cat.*.jpg")
x_train = []

for i in tqdm(dog_train_list):
    temp = img.open(i).resize((64, 64))
    temp = temp.convert("L")
    
    x_train.append((np.array(temp) - np.mean(temp)) / np.std(temp))
    x_train.append((np.array(temp.rotate(90)) - np.mean(temp)) / np.std(temp))
    x_train.append((np.array(temp.rotate(180)) - np.mean(temp)) / np.std(temp))
    x_train.append((np.array(temp.rotate(270)) - np.mean(temp)) / np.std(temp))
    
#    if not idx % 200:
#        print(idx)

y_train = np.tile(1, len(dog_train_list)*4)
print("dog's images loading is done")
for i in tqdm(cat_train_list):
    temp = img.open(i).resize((64, 64))
    temp = temp.convert("L")
    
    x_train.append((np.array(temp) - np.mean(temp)) / np.std(temp))
    x_train.append((np.array(temp.rotate(90)) - np.mean(temp)) / np.std(temp))
    x_train.append((np.array(temp.rotate(180)) - np.mean(temp)) / np.std(temp))
    x_train.append((np.array(temp.rotate(270)) - np.mean(temp)) / np.std(temp))
    
y_train = np.concatenate((y_train, np.tile(0, len(cat_train_list)*4))).astype("uint8")
print("cat's images loading is done")
a = np.asarray(x_train)
x_train = a.reshape(a.shape[0], 1, a.shape[1], a.shape[2])
del(a)
x_train.shape
LeakyReLU = keras.layers.LeakyReLU(alpha=0.01)
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=32, kernel_size=(2, 2), input_shape=(1, 64, 64)))
model.add(LeakyReLU)
model.add(keras.layers.Dropout(rate=0.3))

model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3)))
model.add(LeakyReLU)
model.add(keras.layers.Dropout(rate=0.3))

model.add(keras.layers.Conv2D(filters=64, activation="relu", kernel_size=(3, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
model.add(keras.layers.Dropout(rate=0.3))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=12, activation="relu"))
model.add(keras.layers.Dense(units=1, activation="sigmoid"))
model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.binary_crossentropy, metrics=["binary_accuracy"])
model.summary()
model.fit(x=x_train, y=y_train, epochs=10, validation_split=0.1, shuffle=True)
model.save("Dogs_Cats_model_01.h5")
model = keras.models.load_model("Dogs_Cats_model_01.h5")
model.fit(x=x_train, y=y_train, epochs=2, validation_split=0.1, shuffle=True)
len(model.history.history["binary_accuracy"])
np.arange(1, len(model.history.history["binary_accuracy"])+1, 1)
plt.figure(figsize=(20, 7))
plt.subplot(1, 2, 1)
plt.plot(model.history.history["binary_accuracy"])
plt.plot(model.history.history["val_binary_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Val"], loc="upper left")
#plt.xticks(np.arange(0, len(model.history.history["binary_accuracy"]), 1))

plt.xticks(np.arange(len(model.history.history["binary_accuracy"])), np.arange(1, len(model.history.history["binary_accuracy"])+1, 1))

plt.subplot(1, 2, 2)
plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Val"], loc="upper right")
plt.xticks(np.arange(len(model.history.history["loss"])), np.arange(1, len(model.history.history["loss"])+1, 1))
plt.show()
test_list = glob.glob("../input/test1/test1/*.jpg")
x_test = []

for i in tqdm(test_list):
    temp = img.open(i).resize((64, 64))
    temp = temp.convert("L")
    x_test.append((np.array(temp) - np.mean(temp)) / np.std(temp))

print("test images loading is done")
a = np.asarray(x_test)
x_test = a.reshape(a.shape[0], 1, a.shape[1], a.shape[2])
del(a)

result = model.predict(x=x_test)
idx = []
for i in test_list:
    idx.append(i[21:-4])
result = result.reshape(result.shape[0])
result[result>0.5] = 1
result[result<0.5] = 0
submission = {"id": idx, "label": result}
pd.DataFrame(submission).to_csv("submission.csv", index=False)
pd.DataFrame(submission)