import pandas as pd
train_dir = "../input/train_V2.csv"
train_data = pd.read_csv(train_dir)
train_data.head()
columns = list(train_data.columns)
print("Train Data Contains %d columns & %d Rows " % train_data.shape)
print("\n", columns)
for column in columns:
    print(column, train_data[column].nunique())
train_data.isnull().sum()
train_data.dropna(inplace = True)
def clean_data(source1):
    source = source1.copy()
    source = source.drop(['Id', 'groupId', 'matchId'], axis = 1 )
    source = pd.get_dummies(source, columns=['matchType'], drop_first = True)
    return source

train_data = clean_data(train_data)
train_data = train_data.head(100000)
train_data.head()
from sklearn.model_selection import train_test_split

X = train_data.drop(['winPlacePerc'], axis = 1)
y = train_data.winPlacePerc

train_X , test_X, train_y, test_y = train_test_split(X, y, random_state = 1, test_size = 0.2)

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_X.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
my_model = build_model()
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 10000


# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

history = my_model.fit(train_X, train_y, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

#plot_history(history)
loss, mae, mse = my_model.evaluate(test_X, test_y, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} ".format(mae))
test_predictions = my_model.predict(test_X).flatten()

#plt.scatter(test_y, test_predictions)
#plt.xlabel('True Values [MPG]')
#plt.ylabel('Predictions [MPG]')
#plt.axis('equal')
#plt.axis('square')
#plt.xlim([0,plt.xlim()[1]])
#plt.ylim([0,plt.ylim()[1]])
#_ = plt.plot([-100, 100], [-100, 100])
error = test_predictions - test_y
#plt.hist(error, bins = 25)
#plt.xlabel("Prediction Error [MPG]")
#_ = plt.ylabel("Count")