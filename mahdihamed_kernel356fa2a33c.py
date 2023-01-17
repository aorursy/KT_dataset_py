import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import rcParams

%matplotlib inline

rcParams['figure.figsize'] = 10,8

sns.set(style='whitegrid', palette='muted',

        rc={'figure.figsize': (10,5)})

import os

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# print(os.listdir("../input"))
print(os.listdir("../input/homedataset"))
# Load data as Pandas dataframe

train = pd.read_csv('../input/homedataset/trainHome.csv')

test = pd.read_csv('../input/homedataset/testHome.csv')
train.head()
def display_all(train):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(train)



        

display_all(train.describe(include='all').T)
sns.barplot(x='area m2',y='rate(1-10)', data=train, palette='Set1')

plt.xticks(rotation=0)

plt.show()
sns.barplot(x='price (jd)',y='rate(1-10)', data=train, palette='Set3')

plt.xticks(rotation=0)

plt.show()
sns.barplot(x='price (jd)',y='area m2', data=train, palette='Set3')

plt.xticks(rotation=0)

plt.show()
continuous = ['rate(1-10)', 'area m2']



scaler = StandardScaler()



for var in continuous:

    train[var] = train[var].astype('float64')

    train[var] = scaler.fit_transform(train[var].values.reshape(-1, 1))

    

for var in continuous:

    test[var] = test[var].astype('float64')

    test[var] = scaler.fit_transform(test[var].values.reshape(-1, 1))
    display_all(train.describe(include='all').T)
X_train = train[pd.notnull(train['price (jd)'])].drop(['price (jd)'], axis=1)

Y_train = train[pd.notnull(train['price (jd)'])]['price (jd)']

X_test = test[pd.isnull(test['price (jd)'])].drop(['price (jd)'], axis=1)
def create_model(act='relu', opt='Adam'):

    model = Sequential()

    model.add(Dense(150, activation=act, input_dim=X_train.shape[1]))

    for i in range(1,8):

        model.add(Dense(150, activation=act))

        

    model.add(Dense(1))

    model.compile(loss='mse', optimizer=opt, metrics=['mae'])

    return model
model = create_model()

print(model.summary())
training = model.fit(X_train, Y_train, epochs=100, batch_size=2, validation_split=0.2, verbose=0) 

val_mae = np.mean(training.history['val_mae'])

print("\n%s: %.2f%%" % ('val_mae', val_mae*100))
plt.plot(training.history['mae'])

plt.plot(training.history['val_mae'])

plt.title('model MAE')

plt.ylabel('mean absolute error')

plt.xlabel('epoch')

plt.legend(['train', 'validation Test'], loc='upper right')

plt.show()