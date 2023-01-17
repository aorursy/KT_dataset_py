import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

import tqdm

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from bayes_opt import BayesianOptimization



import time

from sklearn.model_selection import RandomizedSearchCV

from skopt import BayesSearchCV

!pip install MulticoreTSNE

from MulticoreTSNE import MulticoreTSNE as TSNE

from sklearn.manifold import TSNE

import plotly.express as px



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

random_state = 42

np.random.seed(random_state)



df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')

submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
print('Train data Info : ' )

df_train.info()



print('\nTest data Info : ')

df_test.info()



print('\nThere is ' + str(df_train.isnull().sum().sum()) + ' missing values in Train data')

print('\nThere is ' + str(df_test.isnull().sum().sum()) + ' missing values in Train data')



print('\nTrain first 5 rows')

display(df_train.head())



print('\nTest first 5 rows')

df_train.head()
labels = sorted(df_train['label'].unique())

fig = plt.figure(figsize = [20,20])

i = 1

for l in labels :

    for data in df_train[df_train['label'] == l].sample(n=10, random_state=random_state).values :

        plt.subplot(10, 10, i)

        label = data[0]

        pixels = data[1:]

        pixels = np.array(pixels, dtype='uint8')



        pixels = pixels.reshape((28, 28))



        plt.title('Label is {label}'.format(label=label))

        plt.imshow(pixels, cmap='gray', aspect='auto')

        i = i+ 1

plt.tight_layout()

plt.show()
plt.figure(figsize=(25, 15))

j = 1

for i in range(10) :

    plt.subplot(10,4,j)

    j +=1

    plt.imshow(df_train[df_train['label'] == i].sample(1).drop(labels = ["label"],axis = 1).values.reshape(28, 28), cmap='gray', interpolation='none')

    plt.title("Digit: {}".format(i))

    plt.subplot(10,4,j)

    j +=1

    pd.DataFrame(df_train[df_train['label'] == i].sample(1).drop(labels = ["label"],axis = 1).values.reshape(28, 28)).sum(axis = 1).plot.area(title = 'Univariation of Horizontal Pixels')

    plt.subplot(10,4,j)

    j +=1

    pd.DataFrame(df_train[df_train['label'] == i].sample(1).drop(labels = ["label"],axis = 1).values.reshape(28, 28)).sum(axis = 0).plot.area(title = 'Univariation of Vertical Pixels')

    plt.subplot(10,4,j)

    j +=1

    plt.hist(df_train[df_train['label'] == i].sample(1).drop(labels = ["label"],axis = 1))

    plt.title("Pixel Value Distribution")

plt.tight_layout()

ax = sns.barplot(x="index", y="label", data=df_train.label.value_counts().to_frame().sort_index().reset_index())

ax.set(xlabel='Count', ylabel='Degits')
X_train = df_train.drop(labels = ["label"],axis = 1)

Y_train = df_train["label"]

Y_train = to_categorical(Y_train, num_classes = 10)

X_train /= 255



X_embedded = TSNE(n_components=3, n_jobs=1).fit_transform(X_train)

tsne_plot = pd.DataFrame(X_embedded)

tsne_plot.columns = ['tsne_axe1','tsne_axe2','tsne_axe3']

tsne_plot['digits'] = df_train["label"].astype('str')

fig = px.scatter_3d(tsne_plot, x='tsne_axe1', y='tsne_axe2', z='tsne_axe3',

              color='digits',  size_max=18)

fig.show()
X_train = df_train.drop(labels = ["label"],axis = 1)

Y_train = df_train["label"]

Y_train = to_categorical(Y_train, num_classes = 10)



X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)

X_test = df_test.values.reshape(df_test.shape[0], 28, 28, 1)



datagen = ImageDataGenerator(rotation_range=10,

                             zoom_range = 0.1,

                             width_shift_range=0.1,

                             height_shift_range=0.1)



dict_ = {}

batchs = datagen.flow(X_train, Y_train, batch_size=1)

for i in range(784) :

    indx = 'pixel' + str(i)

    dict_[indx] = []

dict_['label'] = []

k = 0

counter = 0

for x_batch, y_batch in tqdm.tqdm(datagen.flow(X_train, Y_train, batch_size=1)):

    for i in range(28):

        for j in range (28):         

            indx = 'pixel' + str(k)

            dict_[indx].append(float(x_batch[0][i][j][0]))

            k = k+1

    dict_['label'].append(np.where(y_batch[0] == 1.)[0].item(0))

    k = 0

    counter += 1

    if counter > 10**5:

        break



df_train = pd.DataFrame.from_dict(dict_)

del dict_
labels = sorted(df_train['label'].unique())

fig = plt.figure(figsize = [20,20])

i = 1

for l in labels :

    for data in df_train[df_train['label'] == l].sample(n=10, random_state=random_state).values :

        plt.subplot(10, 10, i)

        label = data[0]

        pixels = data[1:]

        pixels = np.array(pixels, dtype='uint8')



        pixels = pixels.reshape((28, 28))



        plt.title('Label is {label}'.format(label=label))

        plt.imshow(pixels, cmap='gray', aspect='auto')

        i = i+ 1

plt.tight_layout()

plt.show()
ax = sns.barplot(x="index", y="label", data=df_train.label.value_counts().to_frame().sort_index().reset_index())

ax.set(xlabel='Count', ylabel='Degits')
X_train = df_train.drop(labels = ["label"],axis = 1)

Y_train = df_train["label"]

Y_train = to_categorical(Y_train, num_classes = 10)



X_train /= 255



X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_state)
batch_size = [256, 512]

epochs = [10, 20]

optimizer = ['RMSprop', 'Adam']

neurons = [512, 1024]



def create_model(neurons=neurons, epochs=epochs, optimizer=optimizer, batch_size=batch_size):

    model = Sequential()

    model.add(Dense(neurons, input_dim=784, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(neurons, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



    return model



model_BayesianOptimization = KerasClassifier(build_fn=create_model, verbose=0)

model_GridSearch = KerasClassifier(build_fn=create_model, verbose=0)

model_RandomSearch = KerasClassifier(build_fn=create_model, verbose=0)



param_opt = dict(batch_size=batch_size, epochs=epochs, neurons=neurons, optimizer = optimizer)
grid = GridSearchCV(estimator=model_GridSearch, param_grid=param_opt, n_jobs=1, cv=3, verbose = 0)

grid_result = grid.fit(X_train, Y_train)

#it tooked 80.5min
print('according to gridsearch the best parameters are : ')

print('batch_size : ' + str(grid_result.best_params_['batch_size']))

print('epochs : ' + str(grid_result.best_params_['epochs']))

print('neurons : ' + str(grid_result.best_params_['neurons']))

print('optimizer : ' + str(grid_result.best_params_['optimizer']))
batch_size = grid_result.best_params_['batch_size']

epochs = grid_result.best_params_['epochs']

neurons = grid_result.best_params_['neurons']

optimizer = grid_result.best_params_['optimizer']





model = Sequential()

model.add(Dense(neurons, input_dim=784, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(neurons, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose = 0)



predictions = model.predict_classes(df_test/255)

submission['Label'] = pd.DataFrame(predictions)[0].values

submission.to_csv('submission_grid_search.csv', index=False) 
random = RandomizedSearchCV(estimator=model_RandomSearch, param_distributions=param_opt, n_jobs=1, cv=3, verbose = 1, n_iter = 5)



random_result = random.fit(X_train, Y_train)
print('according to randomsearch the best parameters are : ')

print('batch_size : ' + str(random_result.best_params_['batch_size']))

print('epochs : ' + str(random_result.best_params_['epochs']))

print('neurons : ' + str(random_result.best_params_['neurons']))

print('optimizer : ' + str(random_result.best_params_['optimizer']))
batch_size = random_result.best_params_['batch_size']

epochs = random_result.best_params_['epochs']

neurons = random_result.best_params_['neurons']

optimizer = random_result.best_params_['optimizer']





model = Sequential()

model.add(Dense(neurons, input_dim=784, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(neurons, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose = 2)



predictions = model.predict_classes(df_test/255)

submission['Label'] = pd.DataFrame(predictions)[0].values

submission.to_csv('submission_Random_Search.csv', index=False) 
bayesian = BayesSearchCV(

     model_BayesianOptimization,

     param_opt,

     n_iter=32,

     random_state=random_state,

     cv=3,

     verbose = 0, n_jobs=1

 )



bayesian_result = bayesian.fit(X_train, Y_train)
print('according to BayesianOptimization the best parameters are : ')

print('batch_size : ' + str(bayesian_result.best_params_['batch_size']))

print('epochs : ' + str(bayesian_result.best_params_['epochs']))

print('neurons : ' + str(bayesian_result.best_params_['neurons']))

print('optimizer : ' + str(bayesian_result.best_params_['optimizer']))
batch_size = bayesian_result.best_params_['batch_size']

epochs = bayesian_result.best_params_['epochs']

neurons = bayesian_result.best_params_['neurons']

optimizer = bayesian_result.best_params_['optimizer']





model = Sequential()

model.add(Dense(neurons, input_dim=784, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(neurons, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose = 2)



predictions = model.predict_classes(df_test/255)

submission['Label'] = pd.DataFrame(predictions)[0].values

submission.to_csv('submission_Bayesian_Optimization.csv', index=False) 