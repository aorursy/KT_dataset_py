import matplotlib.pyplot as plt
import plotly.plotly as py
import matplotlib.mlab as mlab
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
credits = pd.read_csv('../input/tmdb_5000_credits.csv', sep=',')
movies = pd.read_csv('../input/tmdb_5000_movies.csv', sep=',')
credits.head()
movies.head(2)
movies[['budget','revenue','vote_count','popularity']].max(axis=0)
hist = movies.hist(bins=50, figsize = (15,10),
                   facecolor='skyblue', ec="darkblue", alpha=0.75)
hist = movies.hist(column=["vote_average"], bins=50, 
                   facecolor='skyblue', ec="darkblue", alpha=0.75)
sns.distplot(movies['vote_average'], hist=True, kde=True, 
             color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1});
n, bins, patches = plt.hist(movies['vote_average'], 
                            50, normed=1, 
                            facecolor='skyblue', ec="darkblue", alpha=0.75)

sigma = movies['vote_average'].std(axis=0)
mu =  movies['vote_average'].mean(axis=0)
plt.plot(bins, mlab.normpdf(bins, mu, sigma),
             'r--', linewidth=2)

plt.xlabel('Vote Average')
plt.ylabel('Probability')
plt.title(r'$\mu=%.2f,\ \sigma=%.2f,\ N=%i$' %(mu, sigma, bins.shape[0]-1))
plt.show()
hist = movies.hist(column=["popularity","budget"], bins=50, figsize = (15,3),
                   facecolor='skyblue', ec="darkblue", alpha=0.75)
hist = movies \
.loc[(movies['popularity']<100) & (movies['budget']<0.3E9)] \
.hist(
    column=["popularity","budget"], 
    bins=50, figsize = (15,3),
    facecolor='skyblue', ec="darkblue", alpha=0.75)
g = sns.JointGrid(x="popularity", y="budget", data=movies)
g = g.plot(sns.regplot, sns.distplot)
g = sns.JointGrid(x="popularity", y="budget", data=movies.loc[(movies['popularity']<100) & (movies['budget']<0.3E9)])
g = g.plot(sns.regplot, sns.distplot)
split = 0.8
msk = np.random.rand(len(movies)) < split

movies_shuff = movies.sample(frac=1).reset_index(drop=True)

train_labels = movies_shuff.loc[msk, movies_shuff.columns =='popularity']
train_data   = movies_shuff.loc[msk, movies_shuff.columns =='budget']

test_labels = movies_shuff.loc[~msk, movies_shuff.columns =='popularity']
test_data   = movies_shuff.loc[~msk, movies_shuff.columns =='budget']
mean = train_data.mean(axis=0)
std  = train_data.std(axis=0)

train_data = (train_data - mean) / std
test_data  = (test_data - mean) / std
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model
model = build_model()
model.summary()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=500,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop])
test_predictions = model.predict(test_data)

print(test_predictions[0:5])
test_labels[0:5]
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
plot_history(history)
line = plt.figure();
plt.plot(test_data, test_labels, ".");
plt.plot(test_data, test_predictions, '.');
def build_model_2():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(1)
  ])
  optimizer = tf.train.RMSPropOptimizer(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model_2 = build_model_2()
model_2.summary()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = model_2.fit(train_data, train_labels, epochs=500,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop])
test_predictions_2 = model_2.predict(test_data)
line = plt.figure()
plt.plot(test_data, test_labels, ".")
plt.plot(test_data, test_predictions, '.')
plt.plot(test_data, test_predictions_2, 'o', mfc='none', markersize=10);
def build_model_3():
  model = keras.Sequential([
    keras.layers.Dense(32, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(1)
  ])
  optimizer = tf.train.RMSPropOptimizer(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model_3 = build_model_3()
model_3.summary()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = model_3.fit(train_data, train_labels, epochs=500,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop])
test_predictions_3 = model_3.predict(test_data)
line = plt.figure()
plt.xlabel('budget')
plt.ylabel('popularity')
plt.plot(test_data, test_labels, ".")
plt.plot(test_data, test_predictions, '.')
plt.plot(test_data, test_predictions_2, 'o', mfc='none', markersize=10)
plt.plot(test_data, test_predictions_3, 'x', mfc='none', markersize=10);