# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.regularizers import l2

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from keras.models import Sequential

from keras.layers import Dense

import numpy

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV
##########################

##Load data

#########################

df = pd.read_csv("/kaggle/input/credit-card-dataset/cc_approvals.data", header = None)

df.info()

print(df.shape)

print(df.head())
##########################

#data preprocessing

##########################

df[15] = df[15].replace(['+'], 1)

df[15] = df[15].replace(['-'], 0)

print("Positive Cases = ", pd.to_numeric(df[15]).sum())



df = df.replace(['?'], np.nan)

print("Shape with null values: ", df.shape)



df = df.dropna()

print("Positive Cases without null values: ", pd.to_numeric(df[15]).sum())

print("Shape without null values: ", df.shape)



print("======================Unique Values in Each Column ======================")

print(df[0].unique())

print(df[3].unique())

print(df[4].unique())

print(df[5].unique())

print(df[6].unique())

print(df[8].unique())

print(df[9].unique())

print(df[10].unique())

print(df[11].unique())

print(df[12].unique())

print("========================================================================")



df[0] = df[0].astype('category')

df[1] = df[1].astype('float64')

df[2] = df[2].astype('float64')

df[3] = df[3].astype('category')

df[4] = df[4].astype('category')

df[5] = df[5].astype('category')

df[6] = df[6].astype('category')

df[7] = df[7].astype('float64')

df[8] = df[8].astype('category')

df[9] = df[9].astype('category')

df[10] = df[10].astype('category')

df[11] = df[11].astype('category')

df[12] = df[12].astype('category')

df[13] = df[13].astype('int64')

df[14] = df[14].astype('int64')



print("======================Data Frame Info Summary ======================")

df.info()

print("====================================================================")



df[0] = df[0].cat.codes

df[3] = df[3].cat.codes

df[4] = df[4].cat.codes

df[5] = df[5].cat.codes

df[6] = df[6].cat.codes

df[8] = df[8].cat.codes

df[9] = df[9].cat.codes

df[10] = df[10].cat.codes

df[11] = df[11].cat.codes

df[12] = df[12].cat.codes



print("======================Data Frame Info Summary ======================")

df.info()

print("====================================================================")



training_data = df.drop(columns=[15])

labels =  pd.DataFrame(df[[15]])



print("==============================Training Data Head===================")

print(training_data.head())

print("====================================================================")

print("=====================================Labels Head===================")

print(labels.head())

print("====================================================================")
##########################

#data normalization

##########################

normalized_training_data=(training_data-training_data.min())/(training_data.max()-training_data.min())

print(normalized_training_data)

print("Positive Cases = ", pd.to_numeric(labels[15]).sum())





training_data_array = np.array(normalized_training_data)

labels_array = np.array(labels)



#print(training_data_array)

#print(labels_array)
#creat model for grid search

def create_model(number_of_neurons,number_of_hidden_layers,activation,optimizer, loss):

    model = Sequential()

    

    for i in range(0,number_of_hidden_layers):

        model.add(Dense(number_of_neurons, input_dim=15, activation=activation))

    

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    

    return model
#hyper parameter optimization using grid search

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=100)



# define the grid search parameters

parameters = {

  'number_of_neurons':[10,15,20],

  'number_of_hidden_layers':[1,2,3,4],

  'activation':['relu', 'sigmoid'],

  'optimizer':['Adam', 'RMSprop'],

  'loss':['binary_crossentropy','mean_squared_error']  

  }

grid = GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1, cv=5, verbose=True, scoring='f1')

grid_result = grid.fit(training_data,labels)



# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
#Manual k fold cross validation with F1 score calculation

kfold = StratifiedKFold(n_splits=5, shuffle=True)

cvscores = []

f1scores = []

iteration = 1



for train_index, test_index in kfold.split(training_data_array,labels_array):



    # create model

    model = Sequential()

    model.add(Dense(10, input_dim=15, activation='sigmoid'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='Adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



    #separate train data and test data

    x_train,x_test=training_data_array[train_index],training_data_array[test_index]

    y_train,y_test=labels_array[train_index],labels_array[test_index]



    model.fit(x_train, y_train, epochs=1000, batch_size=100,verbose=0)

    scores = model.evaluate(x_test, y_test, verbose=0)

    cvscores.append(scores[1] * 100)



    #calculate F1 score

    y_pred = model.predict(x_test, batch_size=100, verbose=1)

    y_pred = np.where(y_pred > 0.5, 1, 0)

    f1 = f1_score(y_test, y_pred, average='macro')

    f1scores.append(f1)

    

    iteration = iteration + 1

    

print("Accuracy: %.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

print("F1: %.2f%% (+/- %.2f%%)" % (numpy.mean(f1scores), numpy.std(f1scores)))
#training and cross validation with regularisation

kfold = StratifiedKFold(n_splits=5, shuffle=True)

cvscores = []

f1scores = []

iteration = 1



for train_index, test_index in kfold.split(training_data_array,labels_array):



    # create model

    model = Sequential()

    model.add(Dense(10, input_dim=15, activation='sigmoid', activity_regularizer=l2(0.0001)))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='Adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



    #separate train data and test data

    x_train,x_test=training_data_array[train_index],training_data_array[test_index]

    y_train,y_test=labels_array[train_index],labels_array[test_index]



    model.fit(x_train, y_train, epochs=1000, batch_size=100,verbose=0)

    scores = model.evaluate(x_test, y_test, verbose=0)

    cvscores.append(scores[1] * 100)



    #calculate F1 score

    y_pred = model.predict(x_test, batch_size=100, verbose=1)

    y_pred = np.where(y_pred > 0.5, 1, 0)

    f1 = f1_score(y_test, y_pred, average='macro')

    f1scores.append(f1)

    

    iteration = iteration + 1

    

print("Accuracy: %.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

print("F1: %.2f%% (+/- %.2f%%)" % (numpy.mean(f1scores), numpy.std(f1scores)))
