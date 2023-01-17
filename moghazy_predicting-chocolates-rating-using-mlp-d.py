import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import seaborn as sns

import matplotlib.pyplot as plt
dataset = pd.read_csv("/kaggle/input/chocolate-bar-ratings/flavors_of_cacao.csv")
dataset.columns = ['Company', 'Specific_Bean_Origin_nor_Bar_Name',

       'REF', 'Review_Date', 'Cocoa_Percent', 'Company_Location', 'Rating',

       'Bean_Type', 'Broad_Bean_Origin']
dataset.head()
dataset.info()
dataset.describe()
dataset.columns
for column_name in dataset.columns:

    print(column_name + "::  {}".format(len(dataset[column_name].value_counts())))
dataset['Company'].value_counts().hist(bins = 4)
dataset['Specific_Bean_Origin_nor_Bar_Name'].value_counts()[dataset['Specific_Bean_Origin_nor_Bar_Name'].value_counts() > 10]
dataset['Specific_Bean_Origin_nor_Bar_Name'].value_counts().hist(bins = 10)
dataset['Company'].value_counts()[dataset['Company'].value_counts() > 18]
dataset['Cocoa_Percent'] = dataset.Cocoa_Percent.apply(lambda x: float(x[:-1])/ 100)
dataset.Review_Date.value_counts()
dataset.Company_Location.value_counts()[dataset.Company_Location.value_counts() > 30]
for new_column in ['USA', 'France', 'Canada', 'UK', 'Italy', 'Ecuador', 'Australia', 'Belgium', 'Switzerland', 'Germany']:

    dataset[new_column] = 0
dataset.loc[dataset["Company_Location"] == 'U.S.A.', 'USA'] = 1

dataset.loc[dataset["Company_Location"] == 'France', 'France'] = 1

dataset.loc[dataset["Company_Location"] == 'Canada', 'Canada'] = 1

dataset.loc[dataset["Company_Location"] == 'U.K.', "UK"] = 1

dataset.loc[dataset["Company_Location"] == 'Italy', 'Italy'] = 1

dataset.loc[dataset["Company_Location"] == 'Ecuador', 'Ecuador'] = 1

dataset.loc[dataset["Company_Location"] == 'Australia', 'Australia'] = 1

dataset.loc[dataset["Company_Location"] == 'Belgium', 'Belgium'] = 1

dataset.loc[dataset["Company_Location"] == 'Switzerland', 'Switzerland'] = 1

dataset.loc[dataset["Company_Location"] == 'Germany', 'Germany'] = 1
dataset.Bean_Type.value_counts()/1795
corr = dataset.corr()

f, ax = plt.subplots(figsize=(25, 25))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, vmax=.1, center=0,

            square=True, linewidths=.5)
training_testing = dataset[['REF', "Review_Date", "Cocoa_Percent", 'Rating', 'USA', 'France', 'Canada', 'UK', 'Italy', 'Ecuador', 'Australia', 'Belgium', 'Switzerland', 'Germany']]
training_testing
training_testing['Rating']
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(1, figsize=(16, 9))

ax = Axes3D(fig, elev=-130, azim=30)

X_reduced = PCA(n_components=3).fit_transform(training_testing.drop(['Rating'], axis = 1))



ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=training_testing['Rating'].round(),

           cmap=plt.cm.Set1, edgecolor='k', s=70)

ax.set_title("First three PCA directions")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])



plt.show()

print("The number of features in the new subspace is " ,X_reduced.shape[1])
import plotly.graph_objects as go



fig = go.Figure(data=[go.Scatter3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2], mode='markers', marker=dict( size=4, color=training_testing['Rating'].round(), colorscale= "Portland", opacity=0.))])



fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.show()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(

                        training_testing.drop(['Rating'], axis = 1), training_testing['Rating'], test_size=0.2, random_state=42)
x_train.shape
def build_model():

    model = keras.Sequential([

        layers.Dense(64, activation='relu', input_dim=13),

        layers.Dense(64, activation='relu'),

        layers.Dense(125, activation='relu'),

        layers.Dense(1)

        ])



    optimizer = tf.keras.optimizers.RMSprop(0.001)



    model.compile(loss='mse',

                  optimizer=optimizer,

                  metrics=['mae', 'mse'])

    return model
model = build_model()
EPOCHS = 1000

model = build_model()



# The patience parameter is the amount of epochs to check for improvement

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='min')



history = model.fit(x_train, y_train, 

                    epochs=EPOCHS, validation_split = 0.2, verbose=1, 

                    callbacks=[early_stop])
import matplotlib.pyplot as plt



plt.plot(history.history['mse'])

plt.plot(history.history['val_mse'])