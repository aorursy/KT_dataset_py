# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



sns.set_style('darkgrid')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')

df = df.sample(frac=1).reset_index(drop=True) # shuffle data

print(df.shape)

df.head()
df.describe()
df.isna().sum()
fig, axes = plt.subplots(ncols=len(df.columns) // 2, nrows=2, figsize=(30,16))



for i, col in enumerate(df.columns[:len(df.columns) // 2]):

    sns.distplot(df[col], bins=10, rug=True, ax=axes[0][i])

    

for i, col in enumerate(df.columns[len(df.columns) // 2:-1]):

    sns.distplot(df[col], bins=10, rug=True, ax=axes[1][i])
sns.countplot(x="target_class", data=df)
plt.figure(figsize=(10,10))

sns.heatmap(df.corr(), annot=True)
fig, axes = plt.subplots(ncols=8, figsize=(30,8))



for i, col in enumerate(df.columns.drop('target_class')):

    sns.boxplot(x='target_class', y=col, data=df, ax=axes[i])

    

plt.tight_layout()
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Lasso

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error



## not doing any features here, just finding features to remove. So no train test split ##



# set the seed to output consistent results

np.random.seed(0)



X = df.drop('target_class', axis=1)

y = df['target_class']
from scipy.stats import normaltest

# H0: normally distributed

# H1: not normally distributed



k2, p = normaltest(X)

k2, p
scaler = StandardScaler()

X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)



k2, p = normaltest(X_scaled)

k2, p
# another test of normality

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html

from scipy.stats import shapiro



for col in X.columns:

    stat, p = shapiro(X[col])

    print('%s\nstatistics=%.3f, p=%.3f' % (col, stat, p))

    # interpret

    alpha = 0.05

    if p > alpha:

        print('Sample looks Gaussian (fail to reject H0)\n')

    else:

        print('Sample does not look Gaussian (reject H0)\n')
from statsmodels.graphics.gofplots import qqplot

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import FunctionTransformer



X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

X_scaled_mm = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)

X_scaled_rs = pd.DataFrame(RobustScaler().fit_transform(X), columns=X.columns)

X_scaled_qt = pd.DataFrame(QuantileTransformer(n_quantiles=5, random_state=0).fit_transform(X), columns=X.columns)

X_scaled_log = pd.DataFrame(FunctionTransformer(np.log1p, validate=True).fit_transform(X), columns=X.columns)



fig, axes = plt.subplots(ncols=len(X.columns), nrows=6, figsize=(30,32))

for i, col in enumerate(X.columns):

    axes[0][i].set_title(col)

    axes[1][i].set_title('Scaled ' + col)

    axes[2][i].set_title('MM Scaled ' + col)

    axes[3][i].set_title('RS Scaled ' + col)

    axes[4][i].set_title('QT Scaled ' + col)

    axes[5][i].set_title('Log Scaled ' + col)

    qqplot(X[col], line='s', ax=axes[0][i])

    qqplot(X_scaled[col], line='s', ax=axes[1][i])

    qqplot(X_scaled_mm[col], line='s', ax=axes[2][i])

    qqplot(X_scaled_rs[col], line='s', ax=axes[3][i])

    qqplot(X_scaled_qt[col], line='s', ax=axes[4][i])

    qqplot(X_scaled_log[col], line='s', ax=axes[5][i])
# all distribution still look non-normal

# only one that looks possibly normal for a few of the variables is the log-transformation

from scipy.stats import shapiro



for col in X_scaled_log.columns:

    stat, p = shapiro(X_scaled_log[col])

    print('Log Transformation of%s\nstatistics=%.3f, p=%.3f' % (col, stat, p))

    # interpret

    alpha = 0.05

    if p > alpha:

        print('Sample looks Gaussian (fail to reject H0)\n')

    else:

        print('Sample does not look Gaussian (reject H0)\n')
from scipy.stats import zscore



X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

print(X_scaled.shape)



ocols = []

zcols = []

for col in X_scaled.columns:

    ocol = 'outlier ' + col

    ocols.append(ocol)

    X_scaled[ocol] = np.where(zscore(X_scaled[col]) <= -2.5, 1,

                              np.where(zscore(X_scaled[col]) >= 2.5, 1, 0))



X_scaled['outlier'] = X_scaled[ocols].sum(axis=1)

X_scaled.drop(ocols, axis=1, inplace=True)

X_scaled = X_scaled[X_scaled['outlier'] == 0]

X_scaled.drop('outlier', axis=1, inplace=True)



print(X_scaled.shape)



X_scaled.head()
from scipy.stats import shapiro



for col in X_scaled.columns:

    stat, p = shapiro(X_scaled[col])

    print('Log Transformation of%s\nstatistics=%.3f, p=%.3f' % (col, stat, p))

    # interpret

    alpha = 0.05

    if p > alpha:

        print('Sample looks Gaussian (fail to reject H0)\n')

    else:

        print('Sample does not look Gaussian (reject H0)\n')

        

fig, axes = plt.subplots(ncols=len(X.columns), figsize=(30,8))

for i, col in enumerate(X.columns):

    axes[i].set_title(col)

    qqplot(X_scaled[col], line='s', ax=axes[i])
scaler = StandardScaler()



# have to icnlude identifier in front of param followed by two underscores

params={'lasso__alpha': [1e-6, 1e-4, 1e-2, 1e-1, 1, 10, 100]}

lasso = Lasso(random_state=0)



# scale the data then apply the lasso regression

pipe = Pipeline(steps=[('scaler', scaler),

                       ('lasso', lasso)

                        ])



gs_cv = GridSearchCV(pipe, params, cv=5, scoring='neg_mean_squared_error')

gs_cv.fit(X, y)



lasso_best = gs_cv.best_estimator_



print(lasso_best.named_steps['lasso'].coef_)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

# from sklearn.linear_model import SGDClassifier  ## Stochastic Gradient Descent ##

from sklearn.model_selection import StratifiedKFold, cross_val_score  ## use StratifiedKfold to preserve distribution of samples from each fold ##



scaler = StandardScaler()



params={'gb': {

            'gb__learning_rate': [1e-4, 1e-2, 1e-1, 1],

            'gb__max_depth': [2, 3, 4],

            'gb__n_estimators': [500], ## Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance ##

            'gb__random_state': [0],

            'gb__tol': [1e-3], ## lower tolerance to speed up algo ##

            },

        'knn': {

            'knn__n_neighbors': [2, 4, 6, 8],            

            },

        'lr': {

            'lr__solver': ['lbfgs'],

            'lr__random_state': [0],

            },

       }



models = {'knn': KNeighborsClassifier(),

          'lr': LogisticRegression(),

          'gb': GradientBoostingClassifier(),

         }



X = df.drop('target_class', axis=1)

y = df['target_class']



kfold = StratifiedKFold(n_splits=3, random_state=0)



for name, model in models.items():

    # scale the data then apply each model

    pipe = Pipeline(steps=[('scaler', scaler),

                           (name, model)

                            ])



    gs_cv = GridSearchCV(pipe, 

                         params[name], 

                         cv=5, 

                         scoring='accuracy',

                         n_jobs=-1)

    

    results = cross_val_score(gs_cv, X, y, cv=kfold)

    results *= 100

    

    print("Results for {}: {:.3f}% ({:.3f}%) [{:.3f}%, {:.3f}%] accuracy".format(name, 

                                                                                results.mean(),

                                                                                results.std(),

                                                                                results.mean() - results.std(),

                                                                                results.mean() + results.std()

                                                                                ))
## Implement Neural Network too ##

## https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/ ##



from keras.models import Sequential

from keras.layers import Dense as Dense2

from keras.wrappers.scikit_learn import KerasClassifier

from keras import regularizers

from keras import callbacks



from tensorflow.keras.layers import Dense, Flatten, Conv2D

from tensorflow.keras import Model

from tensorflow import keras

from tensorflow.keras import layers

import tensorflow as tf



from sklearn.model_selection import cross_val_score, KFold, train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold, cross_val_score  ## use StratifiedKfold to preserve distribution of samples from each fold ##
def baseline_model():

    ## create NN model ##

    model = Sequential()

    ## add first hidden layer - same as # of features##

    model.add(Dense2(X.shape[1], 

                    input_dim=X.shape[1], 

                    kernel_initializer='normal', 

                    activation='relu'))

    ## single output layer ##

    model.add(Dense2(1, kernel_initializer='normal'))

    ## compile model - adam is fastest optimizer ##

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])

    return model



def larger_model():

    model = Sequential()

    ## 1st hidden layer ##

    model.add(Dense2(X.shape[1], 

                    input_dim=X.shape[1], 

                    kernel_initializer='normal', 

                    activation='relu'))

    ## 2nd hidden layer ##

    model.add(Dense2(max(X.shape[1]-3, 2), 

                    kernel_initializer='normal', 

                    activation='relu'))

    ## output layer ##

    model.add(Dense2(1, kernel_initializer='normal'))

    ## Compile model ##

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])

    return model
num_rows = 1000 # don't use full dataset as it takes too long to train



## baseline model ##

model = KerasClassifier(build_fn=baseline_model, 

                        epochs=50, 

                        batch_size=5, 

                        verbose=0)



## NN need input values b/w [-1, +1] ##

## the sigmoid activation function is rather flat for values who's modulus is large so we often normalise inputs to say between [-1, +1] ##

## Should use `sklearn.preprocessing.MinMaxScaler` instead of `StandardScaler`, however, `StandardScaler` gave better results ##

# scaler = MinMaxScaler()

scaler = StandardScaler()



pipe = Pipeline(steps=[('scaler', scaler),

                       ('model', model)

                      ])



X = df.drop('target_class', axis=1)

y = df['target_class']



kfold = StratifiedKFold(n_splits=5, random_state=0)

results = cross_val_score(pipe, X[:num_rows], y[:num_rows], scoring='accuracy', cv=kfold)

results *= 100



print("Results for {}: {:.3f}% ({:.3f}%) [{:.3f}%, {:.3f}%] accuracy".format('`KerasClassifier` baseline model', 

                                                                            results.mean(),

                                                                            results.std(),

                                                                            results.mean() - results.std(),

                                                                            results.mean() + results.std()

                                                                            ))
num_rows = 1000 # don't use full dataset as it takes too long to train



## larger model ##

model = KerasClassifier(build_fn=larger_model, 

                        epochs=50, 

                        batch_size=5, 

                        verbose=0)



## NN need input values b/w [-1, +1] ##

## the sigmoid activation function is rather flat for values who's modulus is large so we often normalise inputs to say between [-1, +1] ##

## Should use `sklearn.preprocessing.MinMaxScaler` instead of `StandardScaler`, however, `StandardScaler` gave better results ##

# scaler = MinMaxScaler()

scaler = StandardScaler()



pipe = Pipeline(steps=[('scaler', scaler),

                       ('model', model)

                      ])



X = df.drop('target_class', axis=1)

y = df['target_class']



kfold = StratifiedKFold(n_splits=5, random_state=0)

results = cross_val_score(pipe, X[:num_rows], y[:num_rows], scoring='accuracy', cv=kfold)

results *= 100



print("Results for {}: {:.3f}% ({:.3f}%) [{:.3f}%, {:.3f}%] accuracy".format('`KerasClassifier` baseline model', 

                                                                            results.mean(),

                                                                            results.std(),

                                                                            results.mean() - results.std(),

                                                                            results.mean() + results.std()

                                                                            ))
X = df.drop('target_class', axis=1)

y = df['target_class']



X_scaled = scaler.fit_transform(X)
def lrelu_01(x): 

    return tf.nn.leaky_relu(x, alpha=0.01)



def build_nn_model():

    '''

    activation function rankings: elu > leaky relu > relu > tanh > sigmoid

    regularization techniques (avoid overfitting): l1-, l2-norm

                                                   Dropout: at every training step, every neuron has a prob p of being temporarily “dropped out”

                                                            prevents overreliance on small subset of neurons (prevent overfitting)

                                                            (Most popular, more than l1, l2)

    BatchNormalization: address the vanishing/exploding gradients problems. 

                        simply zero-centering and normalizing the inputs, 

                        then scaling and shifting the result using two newparameters per layer (one for scaling, the other for shifting)

    '''

    model = keras.Sequential([

    #         layers.Dense(64, activation=lrelu_01, input_shape=[len(X.keys())]),

    #         layers.Dense(64, activation=lrelu_01),

            layers.Dense(64, 

                         activation='elu', 

                         input_shape=[len(X.keys())],),

    #                      kernel_regularizer=regularizers.l2(0.01),

    #                      activity_regularizer=regularizers.l1(0.01)),

            layers.BatchNormalization(),

            layers.Dropout(0.2),

            layers.Dense(64, activation='elu'),

            layers.BatchNormalization(),

            layers.Dropout(0.2),

            layers.Dense(1)

    ])



    optimizer = tf.keras.optimizers.RMSprop(0.001)



    model.compile(loss='mse',

                optimizer=optimizer,

                metrics=['mae', 'mse', 'accuracy'])

    return model
model = build_nn_model()

model.summary()
EPOCHS = 10



'''

To avoid overfitting the training set, interrupt training when its performance on the validation set starts dropping.

Patience - number of epochs that produced the monitored quantity with no improvement after which training will be stopped.

Reference: https://keras.io/callbacks/

'''

es = callbacks.EarlyStopping(monitor='acc', min_delta=0.001, patience=5,

                             verbose=1, mode='max', baseline=None, restore_best_weights=True)



''' 

Learning Rate -  set too high, training diverges 

              -  set too low, training eventually converges to the optimum, but takes long time

              -  solution: start with high LR, then reduce

Reference: https://keras.io/callbacks/

'''

rlr = callbacks.ReduceLROnPlateau(monitor='acc', factor=0.5,

                                  patience=3, min_lr=1e-4, mode='max', verbose=1)



history = model.fit(X_scaled, 

                    y.values,

                    epochs=EPOCHS, 

                    callbacks=[es],

                    validation_split = 0.2,

                    verbose=1)
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
hist[['acc', 'val_acc']].plot(title='NN Acuuracy Score')
from sklearn.decomposition import PCA

from sklearn.cross_decomposition import PLSRegression



## must normalize data before PCA otherwise first factor will appear to explain majority of variance ##

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



pca = PCA(n_components=X_scaled.shape[1], random_state=0)

pca.fit(X_scaled)

var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)

var = np.append([0], var)

var
plt.ylabel('% Variance Explained')

plt.xlabel('# of Features')

plt.title('PCA Analysis')

plt.ylim(0, 110)

plt.xticks(np.arange(1, X.shape[1]+2, 1))

plt.style.context('seaborn-whitegrid')





plt.plot(var)
scaler = StandardScaler()

model = PLSRegression(n_components=4)



pipe = Pipeline(steps=[('scaler', scaler),

                       ('model', model)])



kfold = StratifiedKFold(n_splits=5, random_state=0)

results = cross_val_score(pipe, X, y, cv=kfold)

results *= 100



print("Results for {}: {:.3f}% ({:.3f}%) [{:.3f}%, {:.3f}%] accuracy".format('PLSRegression',

                                                                            results.mean(),

                                                                            results.std(),

                                                                            results.mean() - results.std(),

                                                                            results.mean() + results.std()

                                                                            ))
for i in range(1, X.shape[1]):

    scaler = StandardScaler()

    model = PLSRegression(n_components=i)



    pipe = Pipeline(steps=[('scaler', scaler),

                           ('model', model)])



    kfold = StratifiedKFold(n_splits=5, random_state=0)

    results = cross_val_score(pipe, X, y, cv=kfold)

    results *= 100

    

    print("Results for {}: {:.3f}% ({:.3f}%) [{:.3f}%, {:.3f}%] accuracy".format(i,

                                                                                results.mean(),

                                                                                results.std(),

                                                                                results.mean() - results.std(),

                                                                                results.mean() + results.std()

                                                                                ))
from sklearn.ensemble import VotingClassifier



# voting = 'soft' --> If ‘hard’, uses predicted class labels for majority rule voting. 

# Else if ‘soft’, predicts the class label based on the argmax of the sums of the predicted probabilities

# This gave the best results, so wrap in a function to use in the resampling methods later

def voting_classifier(X, y):

    v_clf = VotingClassifier(estimators=[

                ('knn', KNeighborsClassifier()),

                ('lr', LogisticRegression()),

                ('gb', GradientBoostingClassifier()),

                ], voting='soft')



    params={

            'model__gb__learning_rate': [1e-4, 1e-2, 1e-1, 1],

            'model__gb__n_estimators': [10, 100, 1000],

            'model__gb__random_state': [0],

            'model__knn__n_neighbors': [2, 4, 6, 8],            

            'model__lr__solver': ['lbfgs'],

            'model__lr__random_state': [0],

           }



    pipe = Pipeline(steps=[('scaler', scaler),

                           ('model', v_clf)

                            ])



    gs_cv = GridSearchCV(pipe, 

                         params, 

                         cv=5, 

                         scoring='accuracy',

                         n_jobs=-1,

                         iid=True)



    kfold = StratifiedKFold(n_splits=3, random_state=0)

    results = cross_val_score(gs_cv, X, y, cv=kfold, scoring='accuracy', n_jobs=-1)

    results *= 100

    

    print("Results for {}: {:.3f}% ({:.3f}%) [{:.3f}%, {:.3f}%] accuracy".format('VotingClassifier',

                                                                                results.mean(),

                                                                                results.std(),

                                                                                results.mean() - results.std(),

                                                                                results.mean() + results.std()

                                                                                ))
X = df.drop('target_class', axis=1)

y = df['target_class']

voting_classifier(X,y)
## Running the voting classifier with `gradient boost` takes too long, so leave it out for the resampling methods. ##

def voting_classifier_small(X, y):

    v_clf = VotingClassifier(estimators=[

                ('knn', KNeighborsClassifier()),

                ('lr', LogisticRegression()),

                ], voting='soft')



    params={

            'model__knn__n_neighbors': [2, 4, 6, 8],            

            'model__lr__solver': ['lbfgs'],

            'model__lr__random_state': [0],

           }



    pipe = Pipeline(steps=[('scaler', scaler),

                           ('model', v_clf)

                            ])



    gs_cv = GridSearchCV(pipe, 

                         params, 

                         cv=5, 

                         scoring='accuracy',

                         n_jobs=-1,

                         iid=True)



    kfold = StratifiedKFold(n_splits=3, random_state=0)

    results = cross_val_score(gs_cv, X, y, cv=kfold, scoring='accuracy', n_jobs=-1)

    results *= 100

    

    print("Results for {}: {:.3f}% ({:.3f}%) [{:.3f}%, {:.3f}%] accuracy".format('VotingClassifier',

                                                                                results.mean(),

                                                                                results.std(),

                                                                                results.mean() - results.std(),

                                                                                results.mean() + results.std()

                                                                                ))
from imblearn.over_sampling import RandomOverSampler



X = df.drop('target_class', axis=1)

y = df['target_class']



ros = RandomOverSampler(random_state=0)

X_resampled, y_resampled = ros.fit_resample(X, y)



voting_classifier_small(X_resampled,y_resampled)
from imblearn.over_sampling import SMOTE



X = df.drop('target_class', axis=1)

y = df['target_class']



smote = SMOTE(random_state=0)

X_resampled, y_resampled = smote.fit_resample(X, y)



voting_classifier_small(X_resampled,y_resampled)