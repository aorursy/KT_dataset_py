import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns 



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

import tensorflow.keras as keras

import tensorflow as tf



np.random.seed(5)

tf.set_random_seed(5)



data_file = '../input/data.csv'

df = pd.read_csv(data_file)

df.head(3)
df.describe()
df.isnull().sum()
df = df.drop(['Unnamed: 32'],axis=1);
# Assign 1 for 'M' and 0 for 'B'

df['diagnosis'] = df['diagnosis'].map(lambda s: 1 if s=='M' else 0).astype(int)
feature_columns = list(df.columns)[2:]



# The following line is just some string-foo to split off the main feature name from the mean, se, and worst.

general_features = ['_'.join(s.split('_')[:-1]) for s in feature_columns[0:10]]



print(general_features)
sns.pairplot(df[['radius_mean','area_mean','perimeter_mean']])
# Feature should be a string.

def select_feature(df, feature):

    related_features = [feature+'_'+suffix for suffix in ['mean','se','worst']]

    return df[related_features]



select_feature(df, 'radius').head()
sns.pairplot(select_feature(df, 'radius'))

sns.pairplot(select_feature(df, 'texture'))

sns.pairplot(select_feature(df, 'fractal_dimension'))
def run_cross_val(X, y, num_folds, num_input_units, num_hidden_units):

    kfold = StratifiedKFold(n_splits=num_folds)

    cross_scores = []

    histories = []

    for train, test in kfold.split(X, y):

        scaler = MinMaxScaler(feature_range=(0,1))



        X_train = scaler.fit_transform(X[train])

        X_test = scaler.transform(X[test])

        y_train = y[train]

        y_test = y[test]



        dnn_model = keras.models.Sequential()



        # Input layer

        dnn_model.add(

            keras.layers.Dense(

                units= num_input_units,

                input_dim = X.shape[1],

                kernel_initializer = 'glorot_uniform',

                bias_initializer = 'zeros',

                activation = 'tanh'

            ))



        # hidden layer

        dnn_model.add(

            keras.layers.Dense(

                units=num_hidden_units,

                input_dim = num_input_units,

                kernel_initializer = 'glorot_uniform',

                bias_initializer = 'zeros',

                activation = 'tanh'

            ))



        #output layer

        dnn_model.add(

            keras.layers.Dense(

                units = 1,

                input_dim = num_hidden_units,

                kernel_initializer = 'glorot_uniform',

                bias_initializer = 'zeros',

                activation = 'sigmoid'

            ))



        sgd_optimizer = keras.optimizers.SGD(lr = 0.0001, decay = 1e-7, momentum = .9)



        dnn_model.compile(optimizer = sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy'])



        history = dnn_model.fit(X_train, y_train, epochs = 400, validation_split=0.33, batch_size = 15, verbose = 0)

        histories.append(history.history) # save the histories for later use



        scores = dnn_model.evaluate(X_test, y_test, verbose = 0)



        print("%s: %.2f%%" % (dnn_model.metrics_names[1], scores[1]*100))



        cross_scores.append(scores[1] * 100)



    print("%.2f%% (+/- %.2f%%)" % (np.mean(cross_scores), np.std(cross_scores)))

    

    return histories, cross_scores

    

    
num_input_units = 20

num_hidden_units = 20



X = df.drop(['id','diagnosis'], axis=1).values[:, 0:10]

y = df['diagnosis'].values



print('Full Accuracy Scores:\n')

full_histories, full_cross_scores = run_cross_val(X, y, 10, num_input_units, num_hidden_units)



X = df.drop(['id','diagnosis','area_mean','perimeter_mean'], axis=1).values[:, 0:8]



print('Ablated Accuracy Scores:\n')

ablated_histories, ablated_cross_scores = run_cross_val(X, y, 10, num_input_units, num_hidden_units)
fold_idx = 0

fig, ax = plt.subplots(2, 2, figsize = (10, 10))

results = [full_histories, ablated_histories]

keywords = ['loss','acc']



for i in range(2):

    for j in range(2):            

            ax[i,j].plot(results[i][fold_idx][keywords[j]])

            ax[i,j].plot(results[i][fold_idx]['val_'+keywords[j]])

            ax[i,j].set_xlabel('Epochs')

            ax[i,j].set_ylabel(keywords[j])

            ax[i,j].legend(['Fold Train', 'Fold Test'], loc = 'lower left')

            if i==0:

                ax[i,j].set_title('Non-Ablated')

            else:

                ax[i,j].set_title('Ablated')

            if j==1:

                ax[i,j].set_ylim((0,1))

                