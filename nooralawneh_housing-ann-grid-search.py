import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import rcParams

%matplotlib inline

rcParams['figure.figsize'] = 10,8

sns.set(style='whitegrid', palette='muted',

        rc={'figure.figsize': (15,10)})

import os





# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import rcParams

%matplotlib inline

rcParams['figure.figsize'] = 10,8

sns.set(style='whitegrid', palette='muted',

        rc={'figure.figsize': (15,10)})

import os



import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.impute import SimpleImputer

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout



from numpy.random import seed

import tensorflow 
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
housing = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
#seperate the feautures from the responses

def split_train_test(data, test_ratio):

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
#print the splited data :

#training dataset , testing dataset

print(len(train_set),"train +", len(test_set),"test")
from sklearn.model_selection import train_test_split

train_set,test_set=train_test_split(housing,test_size=0.3,random_state=42)
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)

housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)





for train_index, test_index in split.split(housing, housing["income_cat"]):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]

    

for set in (strat_train_set, strat_test_set):

   set.drop(["income_cat"], axis=1, inplace=True)


train=strat_train_set.drop("median_house_value",axis=1)

test=strat_train_set["median_house_value"].copy()

df = pd.concat([train, test], axis=1, sort=True)
df.head()




def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)



        

display_all(df.describe(include='all').T)



sns.countplot(x='ocean_proximity', data=df, palette='hls') 

plt.xticks(rotation=45) 

plt.show()

df.head()
continuous = ['population', 'median_income', 'median_house_value', 'latitude', 'longitude','housing_median_age','total_rooms','total_bedrooms','households','housing_median_age']

scaler = MinMaxScaler()



for var in continuous:

    df[var] = df[var].astype('float64')

    df[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))

    
df.drop(['housing_median_age','total_rooms','total_bedrooms','households','housing_median_age','ocean_proximity'],axis=1,inplace=True)
df.head()
display_all(df.describe(include='all').T)
X_train = df[pd.notnull(df['median_house_value'])].drop(['median_house_value'], axis=1)

y_train = df[pd.notnull(df['median_house_value'])]['median_house_value']

X_test = df[pd.isnull(df['median_house_value'])].drop(['median_house_value'], axis=1)
def create_model(lyrs=[300,200,100,60,40,35,20,10], act='relu', opt='Adam', dr=0.0):

    

    # set random seed for reproducibility

    seed(42)

    tensorflow.random.set_seed(42)

    

    model = Sequential()

    

    # create first hidden layer

    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))

    

    # create additional hidden layers

    for i in range(1,len(lyrs)):

        model.add(Dense(lyrs[i], activation=act))

    

    # add dropout, default is none

    model.add(Dropout(dr))

    

    # create output layer

    model.add(Dense(1, activation='sigmoid'))  # output layer

    

    model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['accuracy'])

    

    return model
model = create_model()

print(model.summary())
# train model on full train set, with 80/20 CV split



training = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.3, verbose=0)

val_acc = np.mean(training.history['val_accuracy'])

print("\n%s: %.2f%%" % ('val_accuracy', val_acc*100))



plt.plot(training.history['accuracy'])

plt.plot(training.history['val_accuracy'])

plt.title('accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
# create model

model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=32, verbose=0)



# define the grid search parameters

optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']

param_grid = dict(opt=optimizer)



# search the grid

grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)

grid_result = grid.fit(X_train, y_train)
# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# create final model

model = create_model(lyrs=[4], dr=0.2)



print(model.summary())





# train model on full train set, with 80/20 CV split

training = model.fit(X_train, y_train, epochs=50, batch_size=32, 

                     validation_split=0.3, verbose=0)



# evaluate the model

scores = model.evaluate(X_train, y_train)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



# summarize history for accuracy

plt.plot(training.history['accuracy'])

plt.plot(training.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()