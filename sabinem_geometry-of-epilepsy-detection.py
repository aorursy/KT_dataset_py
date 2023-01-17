# use numpy and pandas

import numpy as np

import pandas as pd



# We need sklearn for preprocessing and for the TSNE Algorithm.

import sklearn

from sklearn.preprocessing import Imputer, scale

from sklearn.manifold import TSNE



# WE employ a random state.

RS = 20150101



# We'll use matplotlib for graphics.

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

%matplotlib inline



# We import seaborn to make nice plots.

import seaborn as sns
# Loading the data

X = pd.read_csv('../input/data.csv')

print("The data has {} observations and {} features".format(X.shape[0], X.shape[1]))
# Are there null values in the dataframe?

cols_null_counts = X.apply(lambda x: sum(x.isnull()))

print('number of columns with null values:', len(cols_null_counts[cols_null_counts != 0]))
# Any non numeric datatypes?

datatypes = X.dtypes

print('datatypes that are used: ', np.unique(datatypes.values))



# only the columns of type object concerns us

print('nr of columns for dtype object: ', len(datatypes.values[datatypes.values == 'object']))

print('Columns of type object are: ', [col for col in X.columns if X[col].dtype == 'object'])

X['Unnamed: 0'].values[:10]
# We drop the 'Unnamed: 0' column, maybe it is some internal adminstrative kind of information?

X.drop('Unnamed: 0', inplace=True, axis=1)



# we transform the target into 0 or 1: 0 for normal brain and 1 for the epileptic seizure

X['y'] = X['y'].apply(lambda x: 1 if x == 1 else 0)



# now we sort for the target

X.sort_values(by='y', inplace=True)



# We split the target off the fetures and store it seperately

y = X['y']

X.drop('y', inplace=True, axis=1)

assert 'y' not in X.columns



# make sure the target is binary now

assert set(y.unique()) == {0, 1}



# we also scale the data

X = scale(X) 
# run the Algorithm

epileptic_proj = TSNE(random_state=RS).fit_transform(X)

epileptic_proj.shape
# building the scatter plot function: the target comes in as color, x is the data

def scatter(x, colors):

    """this function plots the result

    - x is a two dimensional vector

    - colors is a code that tells how to color them: it corresponds to the target

    """

    

    # We choose a color palette with seaborn.

    palette = np.array(sns.color_palette("hls", 2)[::-1])



    # We create a scatter plot.

    f = plt.figure(figsize=(10, 8))

    ax = plt.subplot(aspect='equal')

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,

                    c=palette[colors.astype(np.int)]

                   )

    

    ax.axis('off') # the axis will not be shown

    ax.axis('tight') # makes sure all data is shown

    

    # set title

    plt.title("Epilepsy detection", fontsize=25)

    

    # legend with color patches

    epilepsy_patch = mpatches.Patch(color=palette[1], label='Epileptic Seizure')

    normal_patch = mpatches.Patch(color=palette[0], label='Normal Brain Activity')

    plt.legend(handles=[epilepsy_patch, normal_patch], fontsize=10, loc=4)



    return f, ax, sc



# Now we call the scatter plot function on our data

scatter(epileptic_proj, y)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



def apply_KnearestNeighbor(X):

    # split data into train and test sets

    seed = 7

    test_size = 0.33

    X_train, X_test, y_train, y_test = train_test_split(X, y,

        test_size=test_size, random_state=seed)



    # fit model no training data

    model = KNeighborsClassifier(n_neighbors=5)

    model.fit(X_train, y_train)

    print(model)



    # make predictions for test data

    y_pred = model.predict(X_test)

    predictions = [round(value) for value in y_pred]



    # evaluate predictions

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))



    from sklearn.metrics import roc_auc_score

    roc_auc = roc_auc_score(y_test, predictions)

    print("Area Under the Receiver Operating Characteristic Curve: %.2f%%" % roc_auc)
apply_KnearestNeighbor(X)
apply_KnearestNeighbor(epileptic_proj)