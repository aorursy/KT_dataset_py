# Data manipulation packages import.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Data visualization

import seaborn as sns # More data visualization (pretty)



# Supress warnings.

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# Sklearn and XGBoost import.

from sklearn.decomposition import PCA

from sklearn.linear_model import Lasso, Ridge, LogisticRegression

from xgboost import XGBClassifier, XGBRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score as auc

from sklearn.cluster import AgglomerativeClustering as LinkCluster

from sklearn.cluster import KMeans



# Tensorflow import.

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras import Model



# File Download.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def normalise(data, test_data):

    mean = data.mean()

    std = data.std()

    

    data = (data - mean) / std

    test_data = (test_data - mean) / std

    

    return data, test_data



def prep(data, mean=None, std=None, fancy_nan=False):

    '''

    Separate the data into X and Y values, normalise the X data, and handle NaN values.

    '''

    

    # Handle the NaN values based on the algorithm.

    data = handle_nan(data, fancy=fancy_nan)

    

    # Separate the y-values if the exist.

    if 'y' in data.columns:

        features = data.drop(['y'], axis=1)

        labels = data['y']

    else:

        features = data

    

    if 'y' in data.columns:

        return features, labels

    else:

        return features



def handle_nan(data, fancy=False):

    '''

    Handle NaN values.

    '''

    

    if fancy:  # If we want to use Lasso regression to fill in NaN values.

        # Deal with NaN numbers.

        fill_X_open = data[['d_open_interest', 'transacted_qty', 'opened_position_qty ']].dropna(axis=0)

        fill_X_closed = data[['d_open_interest', 'transacted_qty', 'closed_position_qty']].dropna(axis=0)



        # The scale is a bit off (i.e. max d_open_interestis 1361 vs 42 of opened_position_qty), but it should be okay

        lasso_fill_open = Lasso()

        lasso_fill_open.fit(fill_X_open.drop(['opened_position_qty '], axis=1), fill_X_open['opened_position_qty '])



        lasso_fill_closed = Lasso()

        lasso_fill_closed.fit(fill_X_closed.drop(['closed_position_qty'], axis=1), fill_X_closed['closed_position_qty'])



        # Fill in the NaN values for the training set.

        elements = data.loc[data[['closed_position_qty', 'opened_position_qty ']].isna().any(axis=1), ['d_open_interest', 'transacted_qty']]



        data.loc[elements.index, 'opened_position_qty '] = lasso_fill_open.predict(elements)

        data.loc[elements.index, 'closed_position_qty'] = lasso_fill_closed.predict(elements)

    else:  # If we just want to drop the columns with NaN values.

        data = data.fillna(data.mean())

        pass # data = data.dropna(axis=1)

        

    return data





def pca(data, test_data, run=True, n_components=None):

    '''

    Run PCA on the data set.

    '''

    

    if run:

        # Run PCA on the data set.

        transform = PCA(n_components=n_components)

        transform.fit(data)



        # Transform the data set.

        data = transform.fit_transform(data)

        test_data = transform.fit_transform(test_data)

        

        # Convert data back to DataFrame

        data = pd.DataFrame(data)

        test_data = pd.DataFrame(test_data)

    

    return data, test_data





def logistic(array):

    '''

    Convert raw scores to probabilities.

    '''

    

    array = (array - array.mean()) / array.std()

    return np.exp(array) / (1 + np.exp(array))





def export(predictions, train_data=None, path='preds.csv'):

    '''

    Export a list of predictions to a CSV file.

    '''

    

    # Convert the predictions to the proper format.

    predictions = pd.DataFrame(predictions)

    predictions[1] = predictions.index + train_data.shape[0] if train_data is not None else 0

    predictions = predictions[[1, 0]]

    predictions.columns = ['id', 'Predicted']

    predictions = predictions.set_index(predictions.index + 1)

    

    # Load the predictions to a CSV file.

    predictions.to_csv(path, index=False)
def predict(model, data):

    return model.predict(data)[:, 0]
def add_features(data):

    data.loc[:, 'diff'] = data.loc[:, 'ask1'] - data.loc[:, 'bid1']

    data.loc[:, 'bid1/ask1'] = data.loc[:, 'bid1'] / data.loc[:, 'ask1']

    data.loc[:, 'bid_spread'] = data.loc[:, 'bid1'] / data.loc[:, 'bid5']

    data.loc[:, 'ask_spread'] = data.loc[:, 'ask1'] / data.loc[:, 'ask5']

    data.loc[:, 'imbalance1'] = data.loc[:, 'ask1vol'] / data.loc[:, 'bid1vol']

    data.loc[:, 'imbalance2'] = data.loc[:, 'ask2vol'] / data.loc[:, 'bid2vol']

    data.loc[:, 'imbalance3'] = data.loc[:, 'ask3vol'] / data.loc[:, 'bid3vol']

    data.loc[:, 'imbalance4'] = data.loc[:, 'ask4vol'] / data.loc[:, 'bid4vol']

    data.loc[:, 'imbalance5'] = data.loc[:, 'ask5vol'] / data.loc[:, 'bid5vol']

    data.loc[:, 'spread'] = data.loc[:, 'ask1vol'] - data.loc[:, 'bid1vol']

    data.loc[:, 'imb*spread'] = data.loc[:, 'imbalance1'] * data.loc[:, 'spread']

    data.loc[:, 'askvol*ask_spread'] = data.loc[:, 'ask1vol'] * data.loc[:, 'ask_spread']

    data.loc[:, 'askvol*diff'] = data.loc[:, 'ask1vol'] * data.loc[:, 'diff']

    data.loc[:, 'bidvol*diff'] = data.loc[:, 'bid1vol'] * data.loc[:, 'diff']

    

    return data



def add_clusters(data, test_data, clusters=5):

    # K means clustering.

    for c in clusters:

        print(c)

        cluster = KMeans(n_clusters=c, n_jobs=-1).fit(data)

    

        data.loc[:, f'kmeans/{c}'] = cluster.labels_

        test_data.loc[:, f'kmeans/{c}'] = cluster.predict(test_data)

    

    return data, test_data
CLUSTERS = [10, 15, 20]



testing = False



if testing:

    # Load the data, and split it into training and testing.

    input_data = pd.read_csv('/kaggle/input/caltech-cs155-2020/train.csv').set_index('id')

    input_data = input_data.iloc[np.random.permutation(len(input_data))].reset_index(drop=True)



    # Split the data into training and testing.

    t = 4 * input_data.shape[0] // 5



    train = input_data.iloc[:t]

    test = input_data.iloc[t:]

else:

    train = pd.read_csv('/kaggle/input/caltech-cs155-2020/train.csv').set_index('id')

    test = pd.read_csv('/kaggle/input/caltech-cs155-2020/test.csv').set_index('id')



# Let's add some features!!

train = add_features(train)

test = add_features(test)



# Prep the data and split into X and Y.

mean = train.drop(['y'], axis=1).mean(axis=0)

std = train.drop(['y'], axis=1).std(axis=0)



X, Y = prep(train, mean=mean, std=std)



if testing:

    Xt, Yt = prep(test, mean=mean, std=std)

else:

    Xt = prep(test, mean=mean, std=std)



# Add clusters.

# X, Xt = add_clusters(X, Xt, clusters=CLUSTERS)



# Normalise the data.

X, Xt = normalise(X, Xt)
N_MODELS = 1



# Keep track of predictions for each model.

train_preds = pd.DataFrame()

test_preds = pd.DataFrame()



# Add the XGB models to the ensemble.

for i in range(N_MODELS):

    print(f'XGB {i + 1}')

    # Randomly sample the columns

    cols = np.random.choice(X.columns, size=np.random.randint(3, 7), replace=False)

    

    # Train the model.

    model = XGBRegressor(n_jobs=-1, tree_method='gpu_hist', objective='reg:squarederror')

    model.fit(X[cols], Y)



    # Make the predictions.

    train_preds.loc[:, f'xgb{i}'] = model.predict(X[cols])

    test_preds.loc[:, f'xgb{i}'] = model.predict(Xt[cols])



# Add the Logistic models to the ensemble.

for i in range(N_MODELS):

    print(f'Log {i + 1}')

    # Randomly sample the columns.

    cols = np.random.choice(X.columns, size=np.random.randint(3, 7), replace=False)

    

    # Train the model.

    model = LogisticRegression(C=30)

    model.fit(X[cols], Y)



    # Make the predictions.

    train_preds.loc[:, f'log{i}'] = model.predict_proba(X[cols])[:, 1]

    test_preds.loc[:, f'log{i}'] = model.predict_proba(Xt[cols])[:, 1]

    



# One hot encode the labels so that the probability of each can be attained.

Y_1hc = pd.DataFrame(Y)

Y_1hc.columns = ['1']

Y_1hc['0'] = 1 - Y_1hc['1']  



# Add Neural Nets to the ensemble.

for i in range(1):

    print(f'Neural Net {i + 1}')

    # Create Neural Network.

    model = Sequential()

    model.add(Dense(100, input_dim=X.shape[1], activation='relu'))

    model.add(Dense(100, activation='relu'))

    model.add(Dense(70, activation='relu'))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(30, activation='relu'))

    model.add(Dense(20, activation='relu'))

    model.add(Dense(10, activation='relu'))

    model.add(Dense(5, activation='relu'))

    model.add(Dense(2, activation='softmax'))



    # Train the model.

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC()])

    model.fit(X, Y_1hc, epochs=1, batch_size=32)



    # Make the predictions.

    train_preds.loc[:, f'nn{i}'] = predict(model, X)

    test_preds.loc[:, f'nn{i}'] = predict(model, Xt)



print('Complete.')
# Normalise the pooled predictions.

mean = train_preds.mean()

std = train_preds.std()



train_preds = (train_preds - mean) / std

test_preds = (test_preds - mean) / std



# If we want, we can pass the original data to the nn as well.

original = False

if original:

    train_preds = pd.concat([train_preds, X.reset_index(drop=True)], axis=1)

    test_preds = pd.concat([test_preds, Xt.reset_index(drop=True)], axis=1)
# One hot encode the labels so that the probability of each can be attained.

Y_1hc = pd.DataFrame(Y)

Y_1hc.columns = ['1']

Y_1hc['0'] = 1 - Y_1hc['1']



# Create Neural Network.

nn_model = Sequential()

nn_model.add(Dense(200, input_dim=train_preds.shape[1], activation='relu'))

nn_model.add(Dense(300, activation='relu'))

nn_model.add(Dense(200, activation='relu'))

nn_model.add(Dense(50, activation='relu'))

nn_model.add(Dense(2, activation='softmax'))



# Train the model.

nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC()])

nn_model.fit(train_preds, Y_1hc, epochs=1, batch_size=32)



# In-sample error

preds_in = predict(nn_model, train_preds)

train_acc = auc(Y, preds_in)

print(f'Train Acc: {train_acc:.8f}')



# Out-of-sample error.

preds = predict(nn_model, test_preds)



if testing:

    test_acc = auc(Yt, preds)

    print(f'Test Acc:  {test_acc:.8f}')
if not testing:

    export(preds, X)