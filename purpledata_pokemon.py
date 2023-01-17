# This Notebook was created by Krzysztof Kramarz and Damian Kucharski. 

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

Pokemon = pd.read_csv("../input/pokemon/Pokemon.csv")
Pokemon.head()
Pokemon.describe()
from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC 
legendaries = Pokemon[Pokemon["Legendary"] == True].sample(frac = 1)

non_legendaries = Pokemon[Pokemon["Legendary"] == False].sample(frac = 1)
legendaries_train, legendaries_val = legendaries[:int(legendaries.shape[0]*0.7)], legendaries[int(legendaries.shape[0]*0.7):]

non_legendaries_train, non_legendaries_val = non_legendaries[:int(non_legendaries.shape[0]*0.7)], non_legendaries[int(non_legendaries.shape[0]*0.7):]
train = pd.concat([legendaries_train, non_legendaries_train], axis = 0).sample(frac = 1)

val = pd.concat([legendaries_val, non_legendaries_val], axis = 0).sample(frac = 1)
# to_vis_x = Pokemon["Total"].to_numpy()

# to_vis_y = Pokemon["Attack"].to_numpy()

# to_vis_z = Pokemon["Defense"].to_numpy()

# fig = plt.figure()

# ax = fig.add_subplot(111, projection='3d')



# ax.scatter(to_vis_x, to_vis_y, to_vis_z)



to_vis_leg = Pokemon[Pokemon['Legendary'] == True].loc[:, ["Total", "Attack"]]

to_vis_not_leg = Pokemon[Pokemon['Legendary'] == False].loc[:, ["Total", "Attack"]]



#ax = fig.add_subplot(111)

# plt.scatter(to_vis_leg["Total"], to_vis_leg["Attack"], marker = 'rx')

plt.scatter(to_vis_leg["Total"], to_vis_leg["Attack"], marker='+')

plt.scatter(to_vis_not_leg["Total"], to_vis_not_leg["Attack"], marker='x')
len(train)
X_train_logistic_1 = train.iloc[:,[4, 6]].to_numpy()

X_val_logistic_1 = val.iloc[:,[4, 6]].to_numpy()

y_train_logistic_1 = train.iloc[:,[-1]].to_numpy().reshape(len(train), )

y_val_logistic_1 = val.iloc[:,[-1]].to_numpy().reshape(len(val), )
# Instance

logit_model = LogisticRegression(penalty = 'l2', C = 10.0, solver = 'newton-cg')

# Fit

logit_model = logit_model.fit(X_train_logistic_1, y_train_logistic_1)
logit_model.score(X_train_logistic_1, y_train_logistic_1)
logit_model.score(X_val_logistic_1, y_val_logistic_1)
predicted = pd.DataFrame(logit_model.predict(X_val_logistic_1), columns = ['predicted']) #, logit_model.predict_proba(X_val_logistic_1), y_val_logistic_1]

predicted['true'] = y_val_logistic_1
predicted[predicted["true"] != predicted['predicted']]
model = LinearSVC(C = 1, penalty = 'l1', dual = False, loss = 'squared_hinge')

model.fit(X_train_logistic_1, y_train_logistic_1)
y_pred = model.predict(X_val_logistic_1)

predicted_svc = pd.DataFrame(y_pred, columns = ['predicted']) #, logit_model.predict_proba(X_val_logistic_1), y_val_logistic_1]

predicted_svc['true'] = y_val_logistic_1
precision = len(predicted_svc[predicted_svc['true'] == predicted_svc['predicted']]) / len(predicted_svc)

precision
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.losses import binary_crossentropy

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Input

from tensorflow.keras.metrics import BinaryCrossentropy
x_train = train.iloc[:,[5, 6, 7, 8]].to_numpy()

x_val = val.iloc[:,[5, 6, 7, 8]].to_numpy()

y_train = train.iloc[:,[-1]].to_numpy().reshape(len(train), )

y_val = val.iloc[:,[-1]].to_numpy().reshape(len(val), )

y_train = np.array([[0,1] if var == False else [1,0] for var in y_train])

y_val = np.array([[0,1] if var == False else [1,0] for var in y_val])
model = Sequential()

model.add(Input(4))

#model.add(Dropout(0.2))

model.add(Dense(16, activation = 'sigmoid'))

model.add(Dense(8, activation = 'sigmoid'))

model.add(Dense(16, activation = 'sigmoid'))

model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = binary_crossentropy, metrics = [BinaryCrossentropy()])
model.fit(x_train, y_train, epochs = 70, validation_data = [x_val, y_val], verbose = 0)
output = np.array([False if x[0] < 0.5 else True for x in model.predict(x_val)])

output