# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
%matplotlib inline
rcParams['figure.figsize'] = 10,8
sns.set(style='whitegrid', palette='muted',
        rc={'figure.figsize': (15,10)})
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from numpy.random import seed
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/california-housing-prices/housing.csv')
train.head()
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)

        
display_all(train.describe(include='all').T)
sns.countplot(x='ocean_proximity',data=train)
plt.show()
sns.countplot(x='housing_median_age',data=train)
plt.xticks(rotation=90)
plt.show()
# convert to cateogry dtype
train['ocean_proximity'] = train['ocean_proximity'].astype('category')
# convert to category codes
train['ocean_proximity'] = train['ocean_proximity'].cat.codes
continuous = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
              'households', 'median_income', 'median_house_value']

scaler = MinMaxScaler()

for var in continuous:
    train[var] = train[var].astype('float64')
    train[var] = scaler.fit_transform(train[var].values.reshape(-1,1))
train.tail()
display_all(train.describe(include='all').T)
e8 = 0.8*(len(train))
display("e8 = "+(str)(e8))
X_train = train[pd.notnull(train['total_bedrooms'])].drop(['median_house_value'], axis=1)[0:(int)(e8)]
y_train = train[pd.notnull(train['total_bedrooms'])]['median_house_value'][0:(int)(e8)]
X_test  = train[pd.notnull(train['total_bedrooms'])].drop(['median_house_value'], axis=1)[(int)(e8):len(train)]
y_test  = train[pd.notnull(train['total_bedrooms'])]['median_house_value'][(int)(e8):len(train)]
display("X train = "+(str)(len(X_train))+", Y train = "+(str)(len(y_train))+", X test = "+(str)(len(X_test)))
display_all(X_train.describe(include='all').T)
X_train.head()
display_all(y_train.describe(include='all').T)
y_train.head()
display_all(X_test.describe(include='all').T)
X_test.head()
def create_model(lyrs=[8], act='relu', opt='Adam', dr=0.0):

    model = Sequential()
    
    # create first hidden layer
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))
    
    # create additional hidden layers
    for i in range(1,len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))
    
    # add dropout, default is none
    model.add(Dropout(dr))
    
    # create output layer
    model.add(Dense(1))  # output layer
    
    model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['accuracy'])
    
    return model
model = create_model()
print(model.summary())
training = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0, validation_data=(X_test, y_test))
val_acc = np.mean(training.history['val_accuracy'])
print("\n%s: %.2f%%" % ('val accuracy',(val_acc*100)))
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
model = KerasClassifier(build_fn=create_model, verbose=0)

batch_size = [16, 32, 64]
epochs = [50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)

# search the grid
grid = GridSearchCV(estimator=model, 
                    param_grid=param_grid,
                    cv=3,
                    verbose=0)  

grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))