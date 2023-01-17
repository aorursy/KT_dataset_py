import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('bmh')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/data.csv",index_col=0)
data.info()
data.head()
data.shape
data.Age.median()
data.Age.mean()
data.Age.max()
data.Age.min()
index_of_oldest_player=data.Age.idxmax()

print(data.Name.iloc[index_of_oldest_player])
correlation=data.corr(method='pearson')

correlation
plt.figure(figsize=(9, 8))

sns.heatmap(correlation)
plt.figure(figsize=(9, 8))

sns.heatmap(correlation[(correlation >= 0.5) | (correlation <= -0.4)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
plt.figure(figsize=(9, 8))

sns.distplot(data.Age)
plt.figure(figsize=(9, 8))

sns.distplot(data.Overall)
data_no_img = data.drop(columns=["Photo", "Flag", "Club Logo", "Real Face"], axis=1)
data_imp = data_no_img.drop(columns=["ID", "Nationality", "Club", "Preferred Foot",

                                     "Loaned From", "Joined", "Jersey Number",

                                     "Contract Valid Until", "Name", "Work Rate",

                                     "Body Type", "Position", "Height", "Weight", 

                                     "LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW",

                                     "LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", 

                                     "RM", "LWB", "LDM", "CDM", "RDM","RWB", "LB",

                                     "LCB", "RCB", "RB", "CB"], axis=1)
data_imp = data_imp.fillna(0)
data_imp.info()
def convert_money(money):

    if money == 0:

        return float(0)

    else:

        money = money[1:] # remove "€" symbol

        unit = money[-1]

        if unit == 'K':

            value = float(money[0:-1])*1000

        elif unit == 'M':

            value = float(money[0:-1])*1000000

        else:

            value = float(money)

        return value
convert_money("€565K")
data_imp["Wage"] = data_imp["Wage"].apply(convert_money)
data_imp["Value"] = data_imp["Value"].apply(convert_money)
data_imp["Release Clause"] = data_imp["Release Clause"].apply(convert_money)
from sklearn.model_selection import train_test_split
y = data_imp["Value"]

x = data_imp.drop("Value", axis=1)
x_norm = (x-x.mean())/x.std()
x_train, x_test = train_test_split(x_norm, test_size=0.2)
y_train, y_test = train_test_split(y, test_size=0.2)
from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten
model = Sequential([

    Dense(43, input_shape=(43, )),

    Activation('relu'),

    Dense(33),

    Activation('relu'),

    Dense(23),

    Activation('relu'),

    Dense(13),

    Activation('relu'),

    Dense(11),

    Activation('relu'),

    Dense(1),

    Activation('relu')

])
print(model.summary())
from keras import optimizers
# For a mean squared error regression problem

adam = optimizers.Adam(lr=0.003)

model.compile(optimizer=adam,

              loss='mse')
# Train the model, iterating on the data in batches of 64 samples

model.fit(x_train, y_train, epochs=50, batch_size=64)
model.evaluate(x_test, y_test, batch_size=64)
x_train.iloc[[0]]
y_train.iloc[[0]]
ex = x_train.iloc[[0]]



model.predict(ex)
test_predictions = model.predict(x_test).flatten()



plt.scatter(y_test, test_predictions)

plt.xlabel('True Values')

plt.ylabel('Predictions')

plt.axis('equal')

plt.xlim(plt.xlim())

plt.ylim(plt.ylim())

plt.plot([-100, 100], [-100, 100])
error = test_predictions - y_test

plt.hist(error, bins = 50)

plt.xlabel("Prediction Error")

plt.ylabel("Count")
from keras.models import load_model
model.save('fifa_model.h5')  # creates a HDF5 file 'my_model.h5'
# returns a compiled model

# identical to the previous one

exa = load_model('fifa_model.h5')
print(os.listdir('../working'))