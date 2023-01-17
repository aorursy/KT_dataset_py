import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf



pd.options.display.max_rows = 10

pd.options.display.float_format = "{:.1f}".format



import warnings

warnings.filterwarnings('ignore')



tf.__version__
df = pd.read_csv('/kaggle/input/imdb1000/imdb_data.csv', sep='\t')

df = df.rename(columns={'User Votes': 'Votes',

                        'Imdb Rating': 'Rating',

                       'Gross(in Million Dollars)': 'Earnings',

                       'Runtime(Minutes)' : 'Runtime'})

#It is very important to normalise the input features in a proper range

#It helps in avoiding very large calculations

df.Votes = df.Votes / 1000000

df.head()
df.describe()
#Correlation between columns to identify best feature for training a model

df.corr()
plt.figure(figsize=(8,6))

plt.title("Analysis of data points Votes Vs Rating")

sns.scatterplot(x=df.Votes, y=df.Rating)

plt.xlabel('User Votes')

plt.ylabel('IMDB Rating')

plt.show()
def build_model(lr):

    #initialise model :: Sequential Model

    model = tf.keras.Sequential()

    

    #Add layers to the model

    model.add(tf.keras.layers.Dense(units=1,

                                   input_shape=(1,)))

    

    #Compile model

    #Configure training to minimize the model's mean squared error.

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr),

#                     optimizer=tf.keras.optimizers.RMSprop(lr=lr),

                 loss="mean_squared_error",

#                  metrics=[tf.keras.metrics.RootMeanSquaredError()]

                 )

    

    return model
def train(model, dataset, features, label, epochs, batch_size):

    #Feeding the model training data

    history = model.fit(x=dataset[features],

                        y=dataset[label],

                        batch_size=batch_size,

                        epochs=epochs)

    

    weight = model.get_weights()[0]

    bias = model.get_weights()[1]

    

    return weight, bias, history
learning_rate = 0.1

epochs = 15

batch_size = len(df)



feature = "Votes"

label = "Rating"



model = build_model(learning_rate)



w, b, hist = train(model, dataset=df, features=feature, label=label, epochs=epochs, batch_size=batch_size)
print(w[0][0])

print(b[0])

predictions = w[0][0] * df.Votes + b[0]



plt.figure(figsize=(8,6))

plt.title("Analysis of trained model and data points")

sns.scatterplot(x=df.Votes, y=df.Rating)

sns.lineplot(x=df.Votes, y=predictions, color='red')

plt.xlabel('User Votes')

plt.ylabel('IMDB Rating')

plt.show()
LOSS = pd.DataFrame(hist.history)['loss']
plt.figure(figsize=(8,6))

plt.plot(LOSS, label='BGD')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()