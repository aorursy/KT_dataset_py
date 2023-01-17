# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected=True)





# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/international-airline-passengers.csv")

data = data.rename(columns={'International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60': 'nbr_passengers'})
print(data.shape)

data.head()
nan_rows = data[data['nbr_passengers'].isnull()]

print('Number of  NaN rows: ', len(nan_rows) )

data = data.drop(nan_rows.index)



data = data['nbr_passengers']


fig = {

    'data': [{

        'type': 'scatter',

        'y': data,

    }],

    'layout': {

        'title': f'Evolution of Number of passengers',

        'yaxis': {'title': 'Number of Passengers'},

        'xaxis': {'title': 'Months'}

    }

}

iplot(fig)
from keras.models import Sequential

from keras.layers.recurrent import LSTM

from keras.layers.core import Dense, Activation, Dropout

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from sklearn.utils import shuffle
raw_data = data.values.astype("float32").reshape(-1,1)
scaler = MinMaxScaler(feature_range = (0, 1))
raw_data.shape
scaled_data = scaler.fit_transform(raw_data)
shifted = pd.concat([pd.DataFrame(scaled_data).shift(2), pd.DataFrame(scaled_data).shift(1), pd.DataFrame(scaled_data).shift(0)], axis=1)

shifted.head()
shifted = shifted.loc[2:,]

shifted.head()
X = shifted.iloc[:,:2].values

Y = shifted.iloc[:,-1:].values
train_size = 0.7

train_size = int(len(X)*train_size)

X_train = X[:train_size]

Y_train = Y[:train_size]

test_size = len(X) - train_size

X_test = X[:test_size]

Y_test = Y[:test_size]
xtrain = X_train.reshape(X_train.shape[0],X_train.shape[1],1)

xtest = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
model = Sequential()

model.add(LSTM(4, activation='relu', input_shape=(xtrain.shape[1], 1)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(xtrain, 

          Y_train, 

              epochs = 100, 

              batch_size = 1, 

              verbose = 2)
preds = model.predict(xtest)
preds_inversed = scaler.inverse_transform(preds)

Y_test_inversed = scaler.inverse_transform(Y_test.reshape(-1,1))
np.sqrt(mean_squared_error(preds_inversed, Y_test_inversed))


fig = {

    'data': [{

        'type': 'scatter',

        'y': pd.Series(preds_inversed[:,0]) ,

        'name': 'Preds'

    },

    {'type': 'scatter',

        'y': pd.Series(Y_test_inversed[:,0]),

         'name': 'Y'

    }],

    'layout': {

        'title': f'Evolution of Number of passengers',

        'yaxis': {'title': 'Number of Passengers'},

        'xaxis': {'title': 'Months'}

    }

}

iplot(fig)