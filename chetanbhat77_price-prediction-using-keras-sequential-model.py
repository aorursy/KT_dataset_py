!pip install git+https://github.com/tensorflow/docs
# Ignore warnings :
import warnings
warnings.filterwarnings('ignore')

# Data Handling
import pandas as pd

# Visualisation
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

# Tensorflow for regression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
df = pd.read_csv('../input/diamonds/diamonds.csv')
diamondData = df.copy()
diamondData.head()
diamondData = diamondData.drop(['Unnamed: 0'], axis=1)
diamondData.describe()
# Remove the rows that have x, y , z equal to zero
diamondData = diamondData[(diamondData[['x', 'y', 'z']] != 0).all(axis=1)]

# Use x,y,z to create a sythetic feature -- volume
diamondData['volume'] = diamondData['x'] * diamondData['y'] *diamondData['z']
# diamondData['carat_volume'] = diamondData['volume'] * diamondData['carat']
# Correlation Map
plt.figure(figsize=(10,10))
corr = diamondData.corr()
sns.heatmap(data=corr, square=True, annot=True, cmap='RdYlGn')
#pairplotting all the features
sns.pairplot(diamondData, diag_kind="kde")
# Check for NaN
diamondData.isnull().sum()
sns.factorplot(x='cut', data=diamondData , kind='count', aspect= 3)
sns.factorplot(x='cut', y='price',data=diamondData , kind='box', aspect= 3)
sns.factorplot(x='color', data=diamondData , kind='count', aspect= 3)
sns.factorplot(x='color', y='price',data=diamondData , kind='box', aspect= 3)
sns.factorplot(x='clarity', data=diamondData , kind='count', aspect= 3)
sns.factorplot(x='clarity', y='price',data=diamondData , kind='box', aspect= 3)
#One-Hot Encoding for Categorical columns
diamondFeaturesDF =  pd.get_dummies(diamondData)

# Dropping depth , table due to lack of corelation with price
# Dropping x, y, z due to having intercorelation with each other
diamondFeaturesDF = diamondFeaturesDF.drop(columns=['depth', 'table', 'x', 'y', 'z'])
diamondFeaturesDF
# Divide the dataset to train and test sets
train_dataset = diamondFeaturesDF.sample(frac=0.8,random_state=0)
test_dataset = diamondFeaturesDF.drop(train_dataset.index)

#Take out the labels
train_labels = train_dataset.pop('price')
test_labels = test_dataset.pop('price')

# Z-score Normalization
train_dataset_normalized = (train_dataset - train_dataset.mean())/train_dataset.std()
test_dataset_normalized = (test_dataset - test_dataset.mean())/test_dataset.std()
train_dataset_normalized.head()
# Build a sequential model
def build_model():
  model = keras.Sequential([
                            layers.Dense(64, activation='relu', input_shape=[len(train_dataset_normalized.keys())]),
                            layers.Dense(64, activation='relu'),
                            layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  
  return model

model = build_model()
model.summary()
# Train the model
EPOCHS = 500

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3000)

history = model.fit(
  train_dataset_normalized, train_labels,
  batch_size=100, epochs=EPOCHS, validation_split = 0.3, verbose=0,
  callbacks=[early_stop, tfdocs.modeling.EpochDots()])
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({'Basic': history}, metric = "mae")
plt.ylabel('MAE [Price]')
plotter.plot({'Basic': history}, metric = "mse")
plt.ylabel('MSE [Price^2]')
loss, mae, mse = model.evaluate(test_dataset_normalized, test_labels, verbose=2)
print("Testing set loss: {:5.2f}".format(loss))
print("Testing set Mean Abs Error: {:5.2f}".format(mae))
print("Testing set Mean Sqr Error: {:5.2f}".format(mse))
test_predictions = model.predict(test_dataset_normalized).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Price]')
plt.ylabel('Predictions [Price]')
lims = [0, 22000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [Price]")
_ = plt.ylabel("Count")
