import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.regularizers import l2
import csv
import random
physical_devices = tf.config.list_physical_devices("GPU")
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
fraud = []
good = []
data = []
with open("/kaggle/input/creditcard.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    reader = list(reader)
    titles = reader[0]
    V1 = titles.index("V1")
    V19 = titles.index("V19")
    Amount = titles.index("Amount")
    Class = titles.index("Class")
    reader.pop(0)
    amounts = []
    for i in reader:
        amounts.append(float(i[Amount]))
    mam = max(amounts)
    for i in reader:
        if(i[Class]=="1"):
            fraud.append(tuple([float(i[j]) for j in range(V1, V19+1)]+[float(i[Amount])/mam]+[float(i[Class])]))
        else:
            good.append(tuple([float(i[j]) for j in range(V1, V19+1)]+[float(i[Amount])/mam]+[float(i[Class])]))
random.shuffle(good)
good = good[:len(fraud)]
data = fraud+good
random.shuffle(data)
inp = Input(shape=(20,))
x = Dense(20, activation="relu")(inp)
x = Dense(16, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x = Dense(8, activation="relu")(x)
x = Dense(8, activation="relu")(x)
x = Dense(8, activation="relu")(x)
x = Dense(4, activation="relu")(x)
x = Dense(2, activation="relu")(x)
out = Dense(1, activation="sigmoid")(x)
model = Model(inp, out)
model.compile(optimizer="nadam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_x, train_y, batch_size=64, epochs=1000, validation_data=(val_x, val_y), shuffle=True)
save_model(model, "detective.h5")
print(model.evaluate(test_x, test_y))
print(sum(test_y)/len(test_y))