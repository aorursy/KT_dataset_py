%matplotlib inline

import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import os

import glob

import math

import itertools

sns.set()

pd.options.mode.chained_assignment = None  # default='warn'

import warnings

warnings.filterwarnings("ignore")
# Define a heatmap function

def halfHeatMap(df, mirror, corrtype):



    # Create Correlation df

    corr = df.corr(method = corrtype)

    # Plot figsize

    fig, ax = plt.subplots(figsize=(15, 15))

    # Generate Color Map

    colormap = sns.diverging_palette(220, 10, as_cmap=True)



    if mirror == True:

        #Generate Heat Map, allow annotations and place floats in map

        sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")



    else:

        # Drop self-correlations

        dropSelf = np.zeros_like(corr)

        dropSelf[np.triu_indices_from(dropSelf)] = True# Generate Color Map

        colormap = sns.diverging_palette(220, 10, as_cmap=True)

        # Generate Heat Map, allow annotations and place floats in map

        sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", mask=dropSelf)

        

    #show plot

    plt.show()

    

def matches_to_array(matches, feature_list):

    match_list_array=[]

    for match in matches:

        X = np.array(match[feature_list].drop("winPlacePerc", axis = 1))

        y = np.array(match.winPlacePerc)

        match_list_array.append((X,y))

    return match_list_array
train = pd.read_csv("../input/train_V2.csv", low_memory = False)

test = pd.read_csv("../input/test_V2.csv", low_memory = False)
# Inspecting all the features available

train.columns
#Inspecting the size of the training dataset

train.shape
# Inspecting the unique values for match type

train.matchType.unique()
# Inspecting the number of unique matches in the training dataset

len(train.matchId.unique())
# Inspecting the number of matches for games that do not belong to the typical game modes

len(train[train.matchType.str.contains("normal")])
len(train[train.matchType.str.contains("flare")])
len(train[train.matchType.str.contains("crash")])
train_standard = train[~(train.matchType.str.contains("flare") | 

            train.matchType.str.contains("normal") | 

            train.matchType.str.contains("crash"))]
train_standard.shape
standard_solo = ["solo", "solo-fpp"]
standard_multi = ["squad", "duo", "squad-fpp", "duo-fpp"]
non_standard_modes = ["normal-squad-fpp", "crashfpp", "flaretpp", "normal-solo-fpp", "flarefpp", "normal-duo-fpp",

                     "flarefpp", "normal-duo-fpp", "normal-duo", "normal-squad", "crashtpp", "normal-solo"]
for gamemode in train_standard.matchType.unique():

    tempdf = train_standard[train_standard.matchType == gamemode]

    print("Game mode : {}, Number of observations - {}, Number of games - {}".format(gamemode, len(tempdf), len(tempdf.matchId.unique())))
train_standard.drop("rankPoints", axis = 1, inplace = True)
train_standard.shape
train_dict = {}

modes = []

for gamemode in train_standard.matchType.unique():

    modes.append(gamemode)

    train_dict[gamemode] = train_standard[train_standard.matchType == gamemode]
modes
solo = train_dict["solo"].reset_index(drop = True)

solofpp = train_dict["solo-fpp"].reset_index(drop = True)
solo.describe().T
solo = solo.drop(["DBNOs", "revives"], axis = 1)
features = list(solo.describe().columns)

features
for feature in features:

    plt.figure(figsize = (15, 15))

    sns.distplot(solo[feature])
halfHeatMap(solo[features], mirror = False, corrtype = "pearson")
clustermap = sns.clustermap(solo[features], figsize = (15, 15), 

                            metric = "correlation", row_cluster = False, 

                            standard_scale = 1, yticklabels = [])
solofpp.describe().T
solofpp = solofpp.dropna()

solofpp.info()
solofpp = solofpp.drop(["DBNOs", "revives"], axis = 1)
for feature in features:

    plt.figure(figsize = (15, 15))

    sns.distplot(solofpp[feature])
halfHeatMap(solofpp[features], mirror = False, corrtype = "pearson")
clustermap = sns.clustermap(solofpp[features], figsize = (15, 15), 

                            metric = "correlation", row_cluster = False, 

                            standard_scale = 1, yticklabels = [])
for feature in features:

    fig, ax = plt.subplots(figsize = (15, 15))

    sns.distplot(solo[feature], ax = ax, label = "solo")

    sns.distplot(solofpp[feature], ax = ax, label = "solofpp")

    ax.legend()
duo = train_dict["duo"].reset_index(drop = True)

duofpp = train_dict["duo-fpp"].reset_index(drop = True)
duo.describe().T
duo.info()
team_features = list(duo.describe().columns)

team_features
for feature in team_features:

    plt.figure(figsize = (15, 15))

    sns.distplot(duo[feature])
halfHeatMap(duo[team_features], mirror = False, corrtype = "pearson")
clustermap = sns.clustermap(duo[team_features], figsize = (15, 15), 

                            metric = "correlation", row_cluster = False, 

                            standard_scale = 1, yticklabels = [])
duofpp.describe().T
duofpp.info()
for feature in team_features:

    plt.figure(figsize = (15, 15))

    sns.distplot(duofpp[feature])
halfHeatMap(duofpp[team_features], mirror = False, corrtype = "pearson")
clustermap = sns.clustermap(duofpp[team_features], figsize = (15, 15), 

                            metric = "correlation", row_cluster = False, 

                            standard_scale = 1, yticklabels = [])
for feature in team_features:

    fig, ax = plt.subplots(figsize = (15, 15))

    sns.distplot(duo[feature], ax = ax, label = "Duo")

    sns.distplot(duofpp[feature], ax = ax, label = "Duo FPP")

    ax.legend()
squad = train_dict["squad"].reset_index(drop = True)

squadfpp = train_dict["squad-fpp"].reset_index(drop = True)
squad.describe().T
squad.info()
for feature in team_features:

    plt.figure(figsize = (15, 15))

    sns.distplot(squad[feature])
halfHeatMap(squad[team_features], mirror = False, corrtype = "pearson")
clustermap = sns.clustermap(squad[team_features], figsize = (15, 15), 

                            metric = "correlation", row_cluster = False, 

                            standard_scale = 1, yticklabels = [])
squadfpp.describe().T
squadfpp.info(null_counts= True)
for feature in team_features:

    plt.figure(figsize = (15, 15))

    sns.distplot(squadfpp[feature])
halfHeatMap(squadfpp[team_features], mirror = False, corrtype = "pearson")
clustermap = sns.clustermap(squadfpp[team_features], figsize = (15, 15), 

                            metric = "correlation", row_cluster = False, 

                            standard_scale = 1, yticklabels = [])
for feature in team_features:

    fig, ax = plt.subplots(figsize = (15, 15))

    sns.distplot(squad[feature], ax = ax, label = "Duo")

    sns.distplot(squadfpp[feature], ax = ax, label = "Duo FPP")

    ax.legend()
features_to_test = ["assists", "boosts", "damageDealt", "heals", "kills", "killStreaks", 

                    "matchDuration", "rideDistance", "walkDistance", "weaponsAcquired"]
for feature in features_to_test:

    fig, ax = plt.subplots(figsize = (15, 10))

    sns.distplot(solo[feature], ax = ax, label = "Solo")

    sns.distplot(duo[feature], ax = ax, label = "Duo")

    sns.distplot(squad[feature], ax = ax, label = "Squad")

    ax.legend()
solodmg = solo["damageDealt"]

duodmg = duo["damageDealt"]
stat, p = stats.mannwhitneyu(solodmg, duodmg)

print("The Mann-Whitney U Statistic is: {}".format(stat))

print("The corresponding p-value is: {}".format(p))
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import ElasticNetCV

from sklearn.metrics import mean_squared_error, roc_auc_score, mean_absolute_error

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Normalizer, normalize
X = np.array(solo[features].drop("winPlacePerc", axis = 1))

y = np.array(solo.winPlacePerc)

feature_list = list(solo[features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = .2, random_state = 42)
print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', y_train.shape)

print('Validation Features Shape:', X_val.shape)

print('Validation Labels Shape:', y_val.shape)

print('Testing Features Shape:', X_test.shape)

print('Testing Labels Shape:', y_test.shape)
from scipy.cluster.hierarchy import linkage, dendrogram
X_t = np.array(solo[features]).T

features = list(solo[features].columns)
X_norm = normalize(X_t)

mergings = linkage(X_norm, method = "complete")

plt.figure(figsize = (15, 10))

dendrogram(mergings, labels = features, leaf_rotation = 90)

plt.show()
elastic = ElasticNetCV(cv = 5, random_state = 42)
elastic.fit(X_train, y_train)
y_pred = elastic.predict(X_val)



print("R^2: {}".format(elastic.score(X_val, y_val)))

mae = mean_absolute_error(y_val, y_pred)

print("Mean Absolute Error: {}".format(mae))
y_pred = elastic.predict(X_test)



print("R^2: {}".format(elastic.score(X_test, y_test)))

mae = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error: {}".format(mae))
# n_estimator tuning

estimator_options = [5, 7, 10, 15, 20]

estimator_oobs = []

estimator_scores = []

estimator_rmses = []

for estimator in estimator_options:

    model = RandomForestRegressor(n_estimators = estimator, random_state = 42, oob_score = True, n_jobs = -1)

    model.fit(X, y)

    y_pred = model.predict(X_val)

    score = model.oob_score_

    estimator_oobs.append(score)

    estimator_scores.append(model.score(X_val, y_val))

    estimator_rmses.append(mean_squared_error(y_val, y_pred))
for i in range(len(estimator_options)):

    print("n_estimator: {}, OOB score: {}, R^2: {}, RMSE: {}".format(estimator_options[i], 

                                                                     estimator_oobs[i], 

                                                                     estimator_scores[i], 

                                                                     estimator_rmses[i]))
# max_depth tuning, with n_estimator = 10

depth_options = [5, 7, 10, 15, 20]

depth_oobs = []

depth_scores = []

depth_rmses = []

for depth in depth_options:

    model = RandomForestRegressor(n_estimators = 10, max_depth = depth, random_state = 42, oob_score = True, n_jobs = -1)

    model.fit(X, y)

    y_pred = model.predict(X_val)

    score = model.oob_score_

    depth_oobs.append(score)

    depth_scores.append(model.score(X_val, y_val))

    depth_rmses.append(mean_squared_error(y_val, y_pred))
for i in range(len(depth_options)):

    print("max_depth: {}, OOB score: {}, R^2: {}, RMSE: {}".format(depth_options[i], 

                                                                   depth_oobs[i], 

                                                                   depth_scores[i], 

                                                                   depth_rmses[i]))
# max_features tuning, with n_estimator = 10 and max_depth = 15

feature_options = [5, 7, 10, 15, 20]

feature_oobs = []

feature_scores = []

feature_rmses = []

for feature in feature_options:

    model = RandomForestRegressor(n_estimators = 10, max_depth = 15, max_features= feature, 

                                  random_state = 42, oob_score = True, n_jobs = -1)

    model.fit(X, y)

    y_pred = model.predict(X_val)

    score = model.oob_score_

    feature_oobs.append(score)

    feature_scores.append(model.score(X_val, y_val))

    feature_rmses.append(mean_squared_error(y_val, y_pred))
for i in range(len(feature_options)):

    print("max_feature: {}, OOB score: {}, R^2: {}, RMSE: {}".format(feature_options[i], 

                                                                     feature_oobs[i], 

                                                                     feature_scores[i], 

                                                                     feature_rmses[i]))
X_test = np.concatenate((X_val, X_test))

y_test = np.concatenate((y_val, y_test))
print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', y_train.shape)

print('Testing Features Shape:', X_test.shape)

print('Testing Labels Shape:', y_test.shape)
rf = RandomForestRegressor(n_estimators = 10, max_depth = 15, max_features = 15, random_state = 42, n_jobs = -1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("R^2: {}".format(rf.score(X_test, y_test)))

mae = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error: {}".format(mae))
#Quantifying feature importance

importances = list(rf.feature_importances_)

feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances)]

feature_importances_sorted = sorted(feature_importances, key = lambda x: x[1], reverse = True)
for pair in feature_importances_sorted:

    print("Variable: {:25} Importance: {}".format(pair[0], pair[1]))
x_vals = list(range(len(importances)))

plt.figure(figsize = (15, 10))

sns.barplot(x_vals, importances, orientation = "vertical")

plt.xticks(x_vals, feature_list, rotation = "vertical")

plt.ylabel("Importances")

plt.xlabel("Feature")

plt.show()
# Construct new regressor only using most important features

rf_pared = RandomForestRegressor(n_estimators = 10, max_depth = 15, random_state = 42, n_jobs = -1)
feature_importances_sorted
important_features = ["walkDistance", "killPlace", "boosts", "numGroups", "matchDuration"]

important_indices = []

for feature in important_features:

    important_indices.append(feature_list.index(feature))
train_important = X_train[:, important_indices]

test_important = X_test[:, important_indices]

rf_pared.fit(train_important, y_train)
predict_important = rf_pared.predict(test_important)
print("R^2: {}".format(rf_pared.score(test_important, y_test)))

mae = mean_absolute_error(y_test, predict_important)

print("Mean Absolute Error: {}".format(mae))
from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping

from keras import optimizers
X = np.array(solo[features].drop("winPlacePerc", axis = 1))

y = np.array(solo.winPlacePerc)

feature_list = list(solo[features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)
n_cols = X_train.shape[1]

early_stopping_monitor = EarlyStopping(patience = 10)
model = Sequential()

model.add(Dense(10, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(1))



model.compile(optimizer = "adam", loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 30, callbacks = [early_stopping_monitor])
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
model = Sequential()

model.add(Dense(200, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(200, activation = 'relu'))

model.add(Dense(1))



model.compile(optimizer = "adam", loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 30, callbacks = [early_stopping_monitor])
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
model = Sequential()

model.add(Dense(200, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(200, activation = 'relu'))

model.add(Dense(1))



adam = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = adam, loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 50, callbacks = [early_stopping_monitor])
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
model = Sequential()

model.add(Dense(10, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(1))



model.compile(optimizer = "adam", loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 30, callbacks = [early_stopping_monitor])
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
model = Sequential()

model.add(Dense(200, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(200, activation = 'relu'))

model.add(Dense(200, activation = 'relu'))

model.add(Dense(200, activation = 'relu'))

model.add(Dense(1))



model.compile(optimizer = "adam", loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 30, callbacks = [early_stopping_monitor])
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
model = Sequential()

model.add(Dense(200, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(200, activation = 'relu'))

model.add(Dense(200, activation = 'relu'))

model.add(Dense(200, activation = 'relu'))

model.add(Dense(1))

model.compile(optimizer = "adam", loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 100, callbacks = [early_stopping_monitor])
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
from keras.callbacks import LambdaCallback

import keras.backend as K



class LRFinder:

    """

    Instantiates a class to help with finding the optimal learning rate for our neural network.

    In addition, plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.

    See for details:

    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0

    """

    def __init__(self, model):

        self.model = model

        self.losses = []

        self.lrs = []

        self.best_loss = 1e9



    def on_batch_end(self, batch, logs):

        # Log the learning rate

        lr = K.get_value(self.model.optimizer.lr)

        self.lrs.append(lr)



        # Log the loss

        loss = logs['loss']

        self.losses.append(loss)



        # Check whether the loss got too large or NaN

        if math.isnan(loss) or loss > self.best_loss * 4:

            self.model.stop_training = True

            return



        if loss < self.best_loss:

            self.best_loss = loss



        # Increase the learning rate for the next batch

        lr *= self.lr_mult

        K.set_value(self.model.optimizer.lr, lr)



    def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1):

        num_batches = epochs * x_train.shape[0] / batch_size

        self.lr_mult = (end_lr / start_lr) ** (1 / num_batches)



        # Save weights into a file

        self.model.save_weights('tmp.h5')



        # Remember the original learning rate

        original_lr = K.get_value(self.model.optimizer.lr)



        # Set the initial learning rate

        K.set_value(self.model.optimizer.lr, start_lr)



        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))



        self.model.fit(x_train, y_train,

                        batch_size=batch_size, epochs=epochs,

                        callbacks=[callback])



        # Restore the weights to the state before model fitting

        self.model.load_weights('tmp.h5')



        # Restore the original learning rate

        K.set_value(self.model.optimizer.lr, original_lr)



    def plot_loss(self, n_skip_beginning=10, n_skip_end=5):

        """

        Plots the loss.

        Parameters:

            n_skip_beginning - number of batches to skip on the left.

            n_skip_end - number of batches to skip on the right.

        """

        plt.ylabel("loss")

        plt.xlabel("learning rate (log scale)")

        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])

        plt.xscale('log')



    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):

        """

        Plots rate of change of the loss function.

        Parameters:

            sma - number of batches for simple moving average to smooth out the curve.

            n_skip_beginning - number of batches to skip on the left.

            n_skip_end - number of batches to skip on the right.

            y_lim - limits for the y axis.

        """

        assert sma >= 1

        derivatives = [0] * sma

        for i in range(sma, len(self.lrs)):

            derivative = (self.losses[i] - self.losses[i - sma]) / sma

            derivatives.append(derivative)



        plt.ylabel("rate of loss change")

        plt.xlabel("learning rate (log scale)")

        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], derivatives[n_skip_beginning:-n_skip_end])

        plt.xscale('log')

        plt.ylim(y_lim)
def determineLearningRate(xtrain,ytrain,xtest,ytest):    

    model = Sequential()

    model.add(Dense(10, activation = 'relu', input_shape = (n_cols, )))

    model.add(Dense(10, activation = 'relu'))

    model.add(Dense(10, activation = 'relu'))

    model.add(Dense(10, activation = 'relu'))

    model.add(Dense(1))



    model.compile(optimizer = "adam", loss = "mean_absolute_error")

    

    lr_finder = LRFinder(model)

    lr_finder.find(xtrain,ytrain, start_lr=1e-8, end_lr=10, batch_size=1000, epochs=100)

    plt.figure(figsize = (15, 15))

    lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)

    plt.show()

    return model

determineLearningRate(X_train, y_train, X_test, y_test)
# testing adam

model = Sequential()

model.add(Dense(150, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(150, activation = 'relu'))

model.add(Dense(150, activation = 'relu'))

model.add(Dense(150, activation = 'relu'))

model.add(Dense(1))



adam = optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = adam, loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 100, callbacks = [early_stopping_monitor])
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
#testing adagrad

model = Sequential()

model.add(Dense(150, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(150, activation = 'relu'))

model.add(Dense(150, activation = 'relu'))

model.add(Dense(150, activation = 'relu'))

model.add(Dense(1))



adagrad = optimizers.Adagrad(lr=1e-5, epsilon=None, decay=0.0)

model.compile(optimizer = adagrad, loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 100, callbacks = [early_stopping_monitor])
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
#testing adadelta

model = Sequential()

model.add(Dense(150, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(150, activation = 'relu'))

model.add(Dense(150, activation = 'relu'))

model.add(Dense(150, activation = 'relu'))

model.add(Dense(1))



adadelta = optimizers.Adadelta(lr=1e-5, rho=0.95, epsilon=None, decay=0.0)

model.compile(optimizer = adadelta, loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 100, callbacks = [early_stopping_monitor])
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
#testing Nadam

model = Sequential()

model.add(Dense(150, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(150, activation = 'relu'))

model.add(Dense(150, activation = 'relu'))

model.add(Dense(150, activation = 'relu'))

model.add(Dense(1))



nadam = optimizers.Nadam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

model.compile(optimizer = nadam, loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 100, callbacks = [early_stopping_monitor])
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
#testing Adamax

model = Sequential()

model.add(Dense(150, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(150, activation = 'relu'))

model.add(Dense(150, activation = 'relu'))

model.add(Dense(150, activation = 'relu'))

model.add(Dense(1))



adamax = optimizers.Adamax(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

model.compile(optimizer = adamax, loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 100, callbacks = [early_stopping_monitor])
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
X = np.array(squad[team_features].drop("winPlacePerc", axis = 1))

y = np.array(squad.winPlacePerc)

feature_list = list(squad[team_features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = .2, random_state = 42)



print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', y_train.shape)

print('Validation Features Shape:', X_val.shape)

print('Validation Labels Shape:', y_val.shape)

print('Testing Features Shape:', X_test.shape)

print('Testing Labels Shape:', y_test.shape)
X_t = np.array(squad[team_features]).T

features = list(squad[team_features].columns)
X_norm = normalize(X_t)

mergings = linkage(X_norm, method = "complete")

plt.figure(figsize = (15, 10))

dendrogram(mergings, labels = features, leaf_rotation = 90)

plt.show()
# n_estimator tuning

estimator_options = [5, 7, 10, 15, 20]

estimator_oobs = []

estimator_scores = []

estimator_rmses = []

for estimator in estimator_options:

    model = RandomForestRegressor(n_estimators = estimator, random_state = 42, oob_score = True, n_jobs = -1)

    model.fit(X, y)

    y_pred = model.predict(X_val)

    score = model.oob_score_

    estimator_oobs.append(score)

    estimator_scores.append(model.score(X_val, y_val))

    estimator_rmses.append(mean_squared_error(y_val, y_pred))



for i in range(len(estimator_options)):

    print("n_estimator: {}, OOB score: {}, R^2: {}, RMSE: {}".format(estimator_options[i], 

                                                                     estimator_oobs[i], 

                                                                     estimator_scores[i], 

                                                                     estimator_rmses[i]))
# max_depth tuning, with n_estimator = 10

depth_options = [5, 7, 10, 15, 20]

depth_oobs = []

depth_scores = []

depth_rmses = []

for depth in depth_options:

    model = RandomForestRegressor(n_estimators = 10, max_depth = depth, random_state = 42, oob_score = True, n_jobs = -1)

    model.fit(X, y)

    y_pred = model.predict(X_val)

    score = model.oob_score_

    depth_oobs.append(score)

    depth_scores.append(model.score(X_val, y_val))

    depth_rmses.append(mean_squared_error(y_val, y_pred))



for i in range(len(depth_options)):

    print("max_depth: {}, OOB score: {}, R^2: {}, RMSE: {}".format(depth_options[i], 

                                                                   depth_oobs[i], 

                                                                   depth_scores[i], 

                                                                   depth_rmses[i]))
# max_features tuning, with n_estimator = 10 and max_depth = 15

feature_options = [5, 7, 10, 15, 20]

feature_oobs = []

feature_scores = []

feature_rmses = []

for feature in feature_options:

    model = RandomForestRegressor(n_estimators = 10, max_depth = 15, max_features= feature, 

                                  random_state = 42, oob_score = True, n_jobs = -1)

    model.fit(X, y)

    y_pred = model.predict(X_val)

    score = model.oob_score_

    feature_oobs.append(score)

    feature_scores.append(model.score(X_val, y_val))

    feature_rmses.append(mean_squared_error(y_val, y_pred))



for i in range(len(feature_options)):

    print("max_feature: {}, OOB score: {}, R^2: {}, RMSE: {}".format(feature_options[i], 

                                                                     feature_oobs[i], 

                                                                     feature_scores[i], 

                                                                     feature_rmses[i]))
X_test = np.concatenate((X_val, X_test))

y_test = np.concatenate((y_val, y_test))



print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', y_train.shape)

print('Testing Features Shape:', X_test.shape)

print('Testing Labels Shape:', y_test.shape)
rf = RandomForestRegressor(n_estimators = 10, max_depth = 15, max_features = 15, random_state = 42, n_jobs = -1)



rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)



print("R^2: {}".format(rf.score(X_test, y_test)))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
#Quantifying feature importance

importances = list(rf.feature_importances_)

feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances)]

feature_importances_sorted = sorted(feature_importances, key = lambda x: x[1], reverse = True)
for pair in feature_importances_sorted:

    print("Variable: {:25} Importance: {}".format(pair[0], pair[1]))
x_vals = list(range(len(importances)))

plt.figure(figsize = (15, 10))

sns.barplot(x_vals, importances, orientation = "vertical")

plt.xticks(x_vals, feature_list, rotation = "vertical")

plt.ylabel("Importances")

plt.xlabel("Feature")

plt.show()
X = np.array(squad[team_features].drop("winPlacePerc", axis = 1))

y = np.array(squad.winPlacePerc)

feature_list = list(squad[team_features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)

n_cols = X_train.shape[1]
#test out nadam with default parameters and increased layer count

model = Sequential()

model.add(Dense(10, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(1))



model.compile(optimizer = "nadam", loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 30, callbacks = [early_stopping_monitor])
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
X = np.array(duo[team_features].drop("winPlacePerc", axis = 1))

y = np.array(duo.winPlacePerc)

feature_list = list(duo[team_features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)



print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', y_train.shape)

print('Testing Features Shape:', X_test.shape)

print('Testing Labels Shape:', y_test.shape)
rf = RandomForestRegressor(n_estimators = 10, max_depth = 15, max_features = 15, random_state = 42, n_jobs = -1)



rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)



print("R^2: {}".format(rf.score(X_test, y_test)))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
X = np.array(duo[team_features].drop("winPlacePerc", axis = 1))

y = np.array(duo.winPlacePerc)

feature_list = list(duo[team_features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)

n_cols = X_train.shape[1]
#test out nadam with default parameters and increased layer count

model = Sequential()

model.add(Dense(10, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(1))



model.compile(optimizer = "nadam", loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 30)
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
features = list(solo.describe().columns)
X = np.array(solofpp[features].drop("winPlacePerc", axis = 1))

y = np.array(solofpp.winPlacePerc)

feature_list = list(solofpp[features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)



print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', y_train.shape)

print('Testing Features Shape:', X_test.shape)

print('Testing Labels Shape:', y_test.shape)
rf = RandomForestRegressor(n_estimators = 10, max_depth = 15, max_features = 15, random_state = 42, n_jobs = -1)



rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)



print("R^2: {}".format(rf.score(X_test, y_test)))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
X = np.array(solofpp[features].drop("winPlacePerc", axis = 1))

y = np.array(solofpp.winPlacePerc)

feature_list = list(solofpp[features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)

n_cols = X_train.shape[1]
#test out nadam with default parameters and increased layer count

model = Sequential()

model.add(Dense(10, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(1))



model.compile(optimizer = "nadam", loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 30)
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
duo_squad = pd.concat([duo, squad])
X = np.array(duo_squad[team_features].drop("winPlacePerc", axis = 1))

y = np.array(duo_squad.winPlacePerc)

feature_list = list(duo_squad[team_features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)



print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', y_train.shape)

print('Testing Features Shape:', X_test.shape)

print('Testing Labels Shape:', y_test.shape)
rf = RandomForestRegressor(n_estimators = 10, max_depth = 15, max_features = 15, random_state = 42, n_jobs = -1)



rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)



print("R^2: {}".format(rf.score(X_test, y_test)))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
X = np.array(duo_squad[team_features].drop("winPlacePerc", axis = 1))

y = np.array(duo_squad.winPlacePerc)

feature_list = list(duo_squad[team_features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)

n_cols = X_train.shape[1]



#test out nadam with default parameters and increased layer count

model = Sequential()

model.add(Dense(10, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(1))



model.compile(optimizer = "nadam", loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 50)
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
combined_solo = pd.concat([solo, solofpp])
X = np.array(combined_solo[features].drop("winPlacePerc", axis = 1))

y = np.array(combined_solo.winPlacePerc)

feature_list = list(combined_solo[features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)



print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', y_train.shape)

print('Testing Features Shape:', X_test.shape)

print('Testing Labels Shape:', y_test.shape)
rf = RandomForestRegressor(n_estimators = 10, max_depth = 15, max_features = 15, random_state = 42, n_jobs = -1)



rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)



print("R^2: {}".format(rf.score(X_test, y_test)))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
X = np.array(combined_solo[features].drop("winPlacePerc", axis = 1))

y = np.array(combined_solo.winPlacePerc)

feature_list = list(combined_solo[features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)

n_cols = X_train.shape[1]



#test out nadam with default parameters and increased layer count

model = Sequential()

model.add(Dense(10, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(1))



model.compile(optimizer = "nadam", loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 100)
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
#testing out dropout

from keras.layers import Dropout



model = Sequential()

model.add(Dense(10, activation = 'relu', input_shape = (n_cols, )))

model.add(Dropout(0.2))

model.add(Dense(10, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(1))



model.compile(optimizer = "nadam", loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 50)
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
non_standard = train.loc[train["matchType"].isin(non_standard_modes)]
non_standard.shape
X = np.array(non_standard[team_features].drop("winPlacePerc", axis = 1))

y = np.array(non_standard.winPlacePerc)

feature_list = list(non_standard[team_features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)



print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', y_train.shape)

print('Testing Features Shape:', X_test.shape)

print('Testing Labels Shape:', y_test.shape)
rf_other = RandomForestRegressor(n_estimators = 10, max_depth = 15, max_features = 15, random_state = 42, n_jobs = -1)



rf_other.fit(X_train, y_train)

y_pred_other = rf_other.predict(X_test)



print("R^2: {}".format(rf_other.score(X_test, y_test)))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred_other)))
solo_games = pd.concat([solo, solofpp])
X = np.array(solo_games[features].drop("winPlacePerc", axis = 1))

y = np.array(solo_games.winPlacePerc)

feature_list = list(solo_games[features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)



print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', y_train.shape)

print('Testing Features Shape:', X_test.shape)

print('Testing Labels Shape:', y_test.shape)
rf_solo = RandomForestRegressor(n_estimators = 10, max_depth = 15, max_features = 15, random_state = 42, n_jobs = -1)



rf_solo.fit(X_train, y_train)

y_pred_solo = rf_solo.predict(X_test)



print("R^2: {}".format(rf_solo.score(X_test, y_test)))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred_solo)))
model = Sequential()

model.add(Dense(10, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(1))



model.compile(optimizer = "nadam", loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 100, callbacks = [early_stopping_monitor])
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
multiplayer_games = pd.concat([squad, squadfpp, duo, duofpp])
X = np.array(multiplayer_games[team_features].drop("winPlacePerc", axis = 1))

y = np.array(multiplayer_games.winPlacePerc)

feature_list = list(multiplayer_games[team_features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)



print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', y_train.shape)

print('Testing Features Shape:', X_test.shape)

print('Testing Labels Shape:', y_test.shape)
rf_multi = RandomForestRegressor(n_estimators = 10, max_depth = 15, max_features = 15, random_state = 42, n_jobs = -1)



rf_multi.fit(X_train, y_train)

y_pred_multi = rf_multi.predict(X_test)



print("R^2: {}".format(rf_multi.score(X_test, y_test)))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred_multi)))
X = np.array(multiplayer_games[team_features].drop("winPlacePerc", axis = 1))

y = np.array(multiplayer_games.winPlacePerc)

feature_list = list(multiplayer_games[team_features].columns.drop("winPlacePerc"))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)

n_cols = X_train.shape[1]



model = Sequential()

model.add(Dense(10, activation = 'relu', input_shape = (n_cols, )))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(1))



model.compile(optimizer = "nadam", loss = "mean_absolute_error")



model.fit(X_train, y_train, validation_split = 0.2, epochs = 100, callbacks = [early_stopping_monitor])
y_pred = model.predict(X_test)

print("Keras Model's Evaluation: {}".format(np.sqrt(model.evaluate(X_test, y_test))))

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
print("Training Data's Shape: {}".format(train.shape))

print("Testing Data's Shape: {}".format(test.shape))
solo_games = pd.concat([solo, solofpp])
X_solo_train = np.array(solo_games[features].drop("winPlacePerc", axis = 1))

y_solo_train = np.array(solo_games.winPlacePerc)

feature_list = list(solo_games[features].columns.drop("winPlacePerc"))



print('Training Features Shape:', X_solo_train.shape)

print('Training Labels Shape:', y_solo_train.shape)
rf_solo_kaggle = RandomForestRegressor(n_estimators = 10, max_depth = 15, max_features = 15, random_state = 42, n_jobs = -1)



rf_solo_kaggle.fit(X_solo_train, y_solo_train)
multiplayer_games = pd.concat([squad, squadfpp, duo, duofpp])
X_multi_train = np.array(multiplayer_games[team_features].drop("winPlacePerc", axis = 1))

y_multi_train = np.array(multiplayer_games.winPlacePerc)

feature_list = list(multiplayer_games[team_features].columns.drop("winPlacePerc"))



print('Training Features Shape:', X_multi_train.shape)

print('Training Labels Shape:', y_multi_train.shape)
rf_multi_kaggle = RandomForestRegressor(n_estimators = 10, max_depth = 15, max_features = 15, random_state = 42, n_jobs = -1)



rf_multi_kaggle.fit(X_multi_train, y_multi_train)
team_features = list(duo.describe().columns)
X_other_train = np.array(non_standard[team_features].drop("winPlacePerc", axis = 1))

y_other_train = np.array(non_standard.winPlacePerc)

feature_list = list(non_standard[team_features].columns.drop("winPlacePerc"))



print('Training Features Shape:', X_other_train.shape)

print('Training Labels Shape:', y_other_train.shape)
rf_other_kaggle = RandomForestRegressor(n_estimators = 10, max_depth = 15, max_features = 15, random_state = 42, n_jobs = -1)



rf_other_kaggle.fit(X_other_train, y_other_train)
solo_standard_kaggle = test.loc[test["matchType"].isin(standard_solo)]

multi_standard_kaggle = test.loc[test["matchType"].isin(standard_multi)]

other_kaggle = test.loc[test["matchType"].isin(non_standard_modes)]
print('Standard Solo Games Shape:', solo_standard_kaggle.shape)

print('Standard Multiplayer Games Shape:', multi_standard_kaggle.shape)

print('Non Standard Games Shape:', other_kaggle.shape)

print('Total Games Shape:', test.shape)
features = list(solo.describe().columns)

team_features = list(duo.describe().columns)

solo_kaggle_features = features

multi_kaggle_features = team_features
solo_kaggle_features.remove("winPlacePerc")

multi_kaggle_features.remove("winPlacePerc")
X_solo_test = np.array(solo_standard_kaggle[solo_kaggle_features])

solo_IDs = np.array(solo_standard_kaggle.Id)

print('Test Solo Kaggle Shape: ', X_solo_test.shape)

print("Solo Kagge IDs Shape: ", solo_IDs.shape)
X_multi_test = np.array(multi_standard_kaggle[multi_kaggle_features])

multi_IDs = np.array(multi_standard_kaggle.Id)

print('Test Solo Kaggle Shape: ', X_multi_test.shape)

print("Solo Kagge IDs Shape: ", multi_IDs.shape)
X_other_test = np.array(other_kaggle[multi_kaggle_features])

other_IDs = np.array(other_kaggle.Id)

print('Test Solo Kaggle Shape: ', X_other_test.shape)

print("Solo Kagge IDs Shape: ", other_IDs.shape)
# Solo Games

kaggle_solo_pred = rf_solo_kaggle.predict(X_solo_test)
# Multiplayer Games

kaggle_multi_pred = rf_multi_kaggle.predict(X_multi_test)
# Other Games

kaggle_other_pred = rf_other_kaggle.predict(X_other_test)
# Concatenating the IDs with the predictions

kaggle_solo = np.stack((solo_IDs, kaggle_solo_pred), axis = 1)

kaggle_multi = np.stack((multi_IDs, kaggle_multi_pred), axis = 1)

kaggle_other = np.stack((other_IDs, kaggle_other_pred), axis = 1)
final_pred_array = np.concatenate((kaggle_solo, kaggle_multi, kaggle_other))
final_pred_dataframe = pd.DataFrame({'Id':final_pred_array[:,0],'winPlacePerc':final_pred_array[:,1]})
final_pred_dataframe.head()
final_pred_dataframe.to_csv("Final Predictions PUBG", sep='\t', index = False)