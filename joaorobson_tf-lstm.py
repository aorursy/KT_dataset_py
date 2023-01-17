!pip install tensorflow==1.13.2
import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt



from datetime import datetime
tf.__version__
df = pd.read_csv("https://raw.githubusercontent.com/joaorobson/data_science/master/tf_lstm_from_scratch/GlobalTemperatures.csv")
df.head()
df.columns
def get_year_and_month(df):

    date_list = df["dt"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d")).to_list()

    date_list = [x.date().timetuple()[:2] for x in date_list]

    

    year_list, day_list = zip(*date_list)

    

    df["Year"] = year_list

    df["Month"] = day_list

    df = df.drop(["dt"], axis=1)

    

    return df
df = get_year_and_month(df)
df.head()
df_train = df[df["Year"] < 2000]
df_test = df[df["Year"] >= 2000]
df_val = df_train[df_train["Year"] < 1895] # 30% do dataset de treino
df_test = df_test.reset_index()

df_val = df_val.reset_index()
df_train.loc[:, "Year"].unique()
df_test.loc[:, "Year"].unique()
df.groupby("Year").mean()
plt.figure(figsize=(10,10))

plt.title("Temperatura média anual em terra (1850-2015)")

plt.ylabel("Temperatura (Celsius)")

plt.xlabel("Anos")

plt.plot(df["Year"].unique(), df.groupby("Year").mean()["LandAverageTemperature"])

plt.show()
plt.figure(figsize=(10,10))

plt.title("Temperatura média, mínima e máxima média por ano em terra (1850-2015)")

plt.ylabel("Temperatura (Celsius)")

plt.xlabel("Anos")

plt.scatter(df["Year"].unique(), df.groupby("Year").mean()["LandMinTemperature"])

plt.scatter(df["Year"].unique(), df.groupby("Year").mean()["LandAverageTemperature"])

plt.scatter(df["Year"].unique(), df.groupby("Year").mean()["LandMaxTemperature"])

plt.legend()

plt.show()
plt.figure(figsize=(10,10))

plt.title("Temperatura média anual em terra e no oceano (1850-2015)")

plt.ylabel("Temperatura (Celsius)")

plt.xlabel("Anos")

plt.plot(df["Year"].unique(), df.groupby("Year").mean()["LandAndOceanAverageTemperature"])

plt.show()
plt.figure(figsize=(10,10))

plt.title("Diferença entre temperatura média anual em terra e no oceano (1850-2015)")

plt.ylabel("Temperatura (Celsius)")

plt.xlabel("Anos")

plt.scatter(df["Year"].unique(), df.groupby("Year").mean()["LandAndOceanAverageTemperature"] - df.groupby("Year").mean()["LandAverageTemperature"])

plt.show()
# Média da diferença entre as médias de temperatura de oceano + terra e terra

(df.groupby("Year").mean()["LandAndOceanAverageTemperature"] - df.groupby("Year").mean()["LandAverageTemperature"]).mean()
colors_dict = {}

distinct_years = df["Year"].unique()

distinct_colors = ['b', 'g', 'y', 'r']

for ix, years in enumerate(range(0, len(distinct_years),50)):

    for year in range(years, years + 50):

        if year < len(distinct_years):

            colors_dict[distinct_years[year]] = distinct_colors[ix]

colors_dict
from matplotlib.lines import Line2D



fig, ax = plt.subplots()

for i in range(len(df)):

    ax.scatter(df.loc[i]["Month"], df.loc[i]["LandAverageTemperature"], color=colors_dict[df.loc[i]["Year"]])

    ax.legend()

    

fig.set_figheight(15)

fig.set_figwidth(15)

    

legend_elements = [Line2D([0], [0], color=distinct_colors[0], lw=4, label='1850-1899'),

                   Line2D([0], [0], color=distinct_colors[1], lw=4, label='1900-1949'),

                   Line2D([0], [0], color=distinct_colors[2], lw=4, label='1950-1999'), 

                   Line2D([0], [0], color=distinct_colors[3], lw=4, label='2000-2015')]



plt.title("Temperaturas em terra em cada mês (1850-2015)")

plt.ylabel("Temperatura (Celsius)")

plt.xlabel("Meses do ano")

plt.legend(handles=legend_elements, loc=1)

plt.xticks(list(range(1, 13)), ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"], rotation=90)

plt.show()
plt.figure(figsize=(10,10))

plt.title("Temperatura média anual em terra e incertezas (1850-2015)")

plt.ylabel("Temperatura (Celsius)")

plt.xlabel("Anos")

avg_line = plt.plot(df["Year"].unique(), df.groupby("Year").mean()["LandAverageTemperature"], label="Média")

max_line = plt.plot(df["Year"].unique(), df.groupby("Year").mean()["LandAverageTemperature"] + df.groupby("Year").mean()["LandAverageTemperatureUncertainty"], label="Máxima")

min_line = plt.plot(df["Year"].unique(), df.groupby("Year").mean()["LandAverageTemperature"] - df.groupby("Year").mean()["LandAverageTemperatureUncertainty"], label="Mínima")

plt.legend(handles=[avg_line[0], max_line[0], min_line[0]])

plt.show()
plt.figure(figsize=(10,10))

plt.title("Incerteza média por ano das medidas feitas em terra (1850-2015)")

plt.ylabel("Incerteza (Celsius)")

plt.xlabel("Anos")

plt.scatter(df["Year"].unique(), df.groupby("Year").mean()["LandAverageTemperatureUncertainty"])

plt.scatter(df["Year"].unique(), df.groupby("Year").mean()["LandAndOceanAverageTemperatureUncertainty"])

plt.legend()

plt.show()
df_train.drop(columns=['LandMinTemperature',

                 'LandMaxTemperature',

                 'LandAverageTemperatureUncertainty',

                 'LandMaxTemperatureUncertainty',

                 'LandMinTemperatureUncertainty',

                 'LandAndOceanAverageTemperature',

                 'LandAndOceanAverageTemperatureUncertainty'], 

        inplace=True)



df_test.drop(columns=['LandMinTemperature',

                 'LandMaxTemperature',

                 'LandAverageTemperatureUncertainty',

                 'LandMaxTemperatureUncertainty',

                 'LandMinTemperatureUncertainty',

                 'LandAndOceanAverageTemperature',

                 'LandAndOceanAverageTemperatureUncertainty'], 

        inplace=True)



df_val.drop(columns=['LandMinTemperature',

                 'LandMaxTemperature',

                 'LandAverageTemperatureUncertainty',

                 'LandMaxTemperatureUncertainty',

                 'LandMinTemperatureUncertainty',

                 'LandAndOceanAverageTemperature',

                 'LandAndOceanAverageTemperatureUncertainty'], 

        inplace=True)
df_train
from keras.preprocessing.sequence import TimeseriesGenerator



look_back = 12



train_generator = TimeseriesGenerator(df_train["LandAverageTemperature"], df_train["LandAverageTemperature"], length=look_back, batch_size=1)     

test_generator = TimeseriesGenerator(df_test["LandAverageTemperature"], df_test["LandAverageTemperature"], length=look_back, batch_size=1)

val_generator = TimeseriesGenerator(df_val["LandAverageTemperature"], df_val["LandAverageTemperature"], length=look_back, batch_size=1)

X_train = np.array([data[0][0] for data in train_generator])

y_train = np.array([data[1] for data in train_generator])



X_test = np.array([data[0][0] for data in test_generator])

y_test = np.array([data[1] for data in test_generator])



X_val = np.array([data[0][0] for data in val_generator])

y_val = np.array([data[1] for data in val_generator])
from tensorflow.contrib import rnn

from sklearn.metrics import r2_score, mean_squared_error
n_features = 12
epochs = 50

n_classes = 1

n_units = 200

batch_size = 20
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_features])

y = tf.placeholder(tf.float32, [None, 1])
layer = {'weights': tf.Variable(tf.truncated_normal([n_units, n_classes], stddev=0.2), name="w_1"), 'bias': tf.Variable(tf.zeros(shape=[n_classes]))}
with tf.name_scope("LSTM"):

    x = tf.split(X, n_features, 1)



    lstm_cell = rnn.BasicLSTMCell(n_units)

    

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)



with tf.name_scope("output"):

    output = tf.matmul(outputs[-1], layer['weights']) + layer['bias']

with tf.name_scope("loss"):

    logit = output

    cost = tf.reduce_mean(tf.keras.losses.MSE(output, y))

    

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

    train_op = optimizer.minimize(cost)



with tf.name_scope("r2"):

    total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))

    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, output)))

    r_squared = tf.subtract(1.0, tf.div(unexplained_error, total_error))
def train():

    with tf.Session() as sess:



        tf.global_variables_initializer().run()

        tf.local_variables_initializer().run()

        

        losses_train = []

        losses_val = []

        

        for epoch in range(epochs):

            epoch_loss = 0



            i = 0

            for i in range(int(len(X_train) / batch_size)):



                start = i

                end = i + batch_size

                

                batch_x = X_train[start:end]

                batch_y = y_train[start:end]

                

                batch_x_val = X_val[start:end]

                batch_y_val = y_val[start:end]



                _, loss_train, r2 = sess.run([train_op, cost, r_squared], feed_dict={X: batch_x, y: batch_y})

                loss_val, r2_val = sess.run([cost, r_squared], feed_dict={X:batch_x_val, y:batch_y_val})

                

                losses_val.append(loss_val)

                losses_train.append(loss_train)

                

                i += batch_size





            print('Epoch', epoch + 1, 'de', epochs)

            print('Treino:\nLoss:', loss_train, 'R2:', r2)

            print('Validação:\nLoss:', loss_val, 'R2:', r2_val)

            print()



        prediction = sess.run(output, feed_dict={X:X_test})        

        return prediction, losses_val, losses_train



y_pred, losses_val, losses_train = train()
plt.figure(figsize=(10,10))

plt.title("Loss - Mean Squared Error")

plt.plot(losses_val, label="Validação")

plt.plot(losses_train, label="Treino")

plt.legend()

plt.show()
print('R2 score:', r2_score(y_test.ravel(), y_pred.ravel()))
print('MSE:', mean_squared_error(y_test.ravel(), y_pred.ravel()))
plt.figure(figsize=(10,10))

plt.title('Dado de teste vs Predição')

plt.plot(y_test, label="Dado de teste")

plt.plot(y_pred, label="Predição")

plt.legend()

plt.show()
plt.figure(figsize=(10,10))

plt.title("Temperaturas médias - Real vs Predita")

plt.plot(np.concatenate([df_train.groupby("Year").mean()["LandAverageTemperature"].values, list(map(np.mean, np.array_split(y_pred.ravel(), 16)))]), label="Predição")

plt.plot(np.concatenate([df_train.groupby("Year").mean()["LandAverageTemperature"].values, list(map(np.mean, np.array_split(y_test.ravel(), 16)))]), label="Dado de teste")

plt.plot(df_train.groupby("Year").mean()["LandAverageTemperature"].values, label="Dado de treino")

plt.legend()

plt.show()