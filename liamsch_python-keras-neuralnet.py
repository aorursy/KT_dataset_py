# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from keras.callbacks import EarlyStopping

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten

from keras.regularizers import l2, activity_l2

from keras.layers.convolutional import Convolution1D, MaxPooling1D

from keras.optimizers import Adam



np.random.seed(1)  # for reproducibility



train_in = pd.read_csv('../input/train.csv')

test_in = pd.read_csv('../input/test.csv')
### Create dummy variables and prepare training/test data frames...



# Save our target variable column before we drop it

y_train = train_in.iloc[:, -1]



# Stack training and test data into one big data frame so we cover all categories found in both

data = pd.concat((train_in.iloc[:, :-1], test_in))



# Get dummy variables for all categorical columns

colnames = []

X_data = []



# For each categorical column...

for c in [i for i in data.columns[1:-1] if 'cat' in i]:

    # Get dummy variables for that column

    dummies = pd.get_dummies(data[c])

    # Drop the last dummy, as its value is implied by the others (all 0's = 1)

    dummies = dummies.iloc[:, :-1].values.astype(np.bool)

    X_data.append(dummies)

    # Create column names for those dummy variables

    colnames += [c + '_' + str(i) for i in range(dummies.shape[1])]

    

# Stack all dummy variables into big dataframe with the colnames

X_data = pd.DataFrame(np.hstack(X_data), columns=colnames)



# Drop any columns with only 1 value (so drop unused categories, if any)

X_data = X_data.iloc[:, [len(pd.unique(X_data.loc[:,c]))>1 for c in X_data.columns]]



# Get the other (continuous) columns

X_data_cont = np.vstack([data[c].values.astype(np.float32) \

                         for c in data.columns[1:-1] if 'cat' not in c]).T



# Final data frame is the dummy variables + the continuous variables

X_data = X_data.join(pd.DataFrame(X_data_cont, 

                    columns=[c for c in data.columns[1:-1] if 'cat' not in c]))



# Split into train and test data frames

train = X_data[:len(y_train)].join(y_train)

test = X_data[len(y_train):]



# Create X train and y train arrays for input to NN

Xt = train.iloc[:, :-1].values

yt = train.iloc[:, -1].values
### Define and compile a fairly deep neural network...

dropout_p=0.5



model = Sequential()

model.add(Dropout(0.1, input_shape=[Xt.shape[1]]))

model.add(Dense(2048))

model.add(Activation('relu'))

model.add(Dropout(dropout_p))

model.add(Dense(1024))

model.add(Activation('relu'))

model.add(Dropout(dropout_p))

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(dropout_p))

model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(dropout_p))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dense(1))

model.summary()

model.compile(loss='mae', optimizer=Adam())
### Fit the neural network model (takes 3-4 mins per epoch, 1 epoch seems to provide best results,

### though I only tried with 1 or 2)

history = model.fit(Xt, yt,

                    batch_size=96,

                    nb_epoch=1,

                    verbose=2, 

                    #validation_data=(Xv.values, yv.values),

                    shuffle=True)
### Predict the test set

# This NN submission got me to 1214 on LB

# Taking logs of the loss before training and then exponentiating the predictions degraded performance

# Running two epochs also degraded performance

preds_nn = model.predict(test.values)

submission_nn = pd.DataFrame(test_in['id'])

submission_nn['loss'] = preds_nn

submission_nn.to_csv('submission_nn.csv', index=False)