import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense



seed = 7 

np.random.seed(seed)
# load data

dataset = pd.read_csv('../input/diabetes.csv')
# convert column names to lower case

dataset.columns = [item.lower() for item in dataset.columns]



# extract data and labels

data_columns = [column for column in dataset.columns if column != 'outcome']

label_column = list(set(dataset.columns) - set(data_columns))



data = dataset[data_columns]

labels = dataset[label_column]

# create model

model = Sequential()

model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))

model.add(Dense(8, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='sigmoid'))





# Compile model

model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])



# Fit the model

X = data.values

Y = labels.values

model.fit(X, Y, nb_epoch=150, batch_size=10)



# evaluate the model

scores = model.evaluate(X, Y)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))