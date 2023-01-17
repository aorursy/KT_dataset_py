import pandas as pd
FILENAME = '/kaggle/input/heart-disease-uci/heart.csv'

dataset = pd.read_csv(FILENAME)
dataset.head(5)
from sklearn.preprocessing import StandardScaler

def pre_process_dataset(filename):
    dataset = pd.read_csv(filename)

    X = dataset.iloc[:,:-1]
    Y = dataset.iloc[:,-1:]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    dataset_columns_name = dataset.columns.tolist()
    X = pd.DataFrame(X, columns=dataset_columns_name[:-1])
    df = X.join(Y)
    df.to_csv("heart_pre.csv")

    return df
pre_processed_dataset = pre_process_dataset(FILENAME)
pre_processed_dataset.head(5)
X = pre_processed_dataset.iloc[:,:-1]
X.head(5)
Y = pre_processed_dataset.iloc[:,-1:]
Y.head(5)
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(X, Y,  test_size=0.30)
x_train.head(5)
y_train.head(5)
import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_shape = [13], units= 8, activation= tf.nn.relu),
    tf.keras.layers.Dense(units= 64, activation= tf.nn.relu),
    tf.keras.layers.Dense(units= 128, activation= tf.nn.relu),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units= 1024, activation= tf.nn.relu),
    tf.keras.layers.Dense(units= 2048, activation= tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units= 4096, activation= tf.nn.relu),
#     tf.keras.layers.Dense(units= 512, activation= tf.nn.relu),
    tf.keras.layers.Dense(units= 8, activation= tf.nn.relu),
    tf.keras.layers.Dense(units= 1, activation= tf.nn.sigmoid),
    
])
model.compile(optimizer = tf.keras.optimizers.Adam(), loss = tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
model_history = model.fit(x_train, y_train, epochs = 100, validation_data = (x_test,y_test))

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if logs.get('loss') < .8 and logs.get('val_loss') < .8 and logs.get('val_accuracy') > .80 and logs.get('accuracy') > .91:
            print ("Reached the limits")
            self.model.stop_training = True
callback = MyCallback()
history = model.fit(x_train, y_train, epochs = 100, validation_data = (x_test,y_test), callbacks=[callback])
model.summary()
