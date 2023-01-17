# use numpy and pandas

import numpy as np

import pandas as pd



# We need sklearn for preprocessing and for the TSNE Algorithm.

import sklearn

from sklearn.preprocessing import Imputer, scale

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



# WE employ a random state.

RS = 20150101



# We'll use matplotlib for graphics.

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

%matplotlib inline



# We import seaborn to make nice plots.

import seaborn as sns
# import the data: we just need the training data for this visualization

# since we need a target!

X = pd.read_csv('../input/train.csv')



# sort by target

X.sort_values(by='Survived', inplace=True)



# separate the target

y = X['Survived']



# transform all fields that are not numeric:

def prepare_for_ml(X):

    # Cabin is 0 if nan or 1 if filled

    X.Cabin = X.Cabin.apply(lambda x: 0 if pd.isnull(x) else 1) 



    # Sex is turned into 0/1 for male/female

    X.Sex = X.Sex.apply(lambda x: 0 if x == 'male' else 1)



    # Embarked is encoded as 1,2,3 maintaining the order of ports S -> C -> Q

    def get_port_nr(embarked):

        if embarked == 'C':

            return 2

        elif embarked == 'Q':

            return 3 

        else: # cases nan or 'S'

            return 1



    X.Embarked = X.Embarked = X.Embarked.apply(lambda x: get_port_nr(x))



    # Name Ticket and PassengerId are dropped

    X.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], inplace=True, axis=1)

    

    print("Features: ", X.columns)



    # now the missing values are imputed, which are Fare and Age, since Embarked has 

    # already been filled!

    imputer = Imputer()

    X = imputer.fit_transform(X)

    

    # scale the feature values 

    X = scale(X)

    

    return X



# apply transformation

X = prepare_for_ml(X)
# run the TSNE Algorithm

titanic_proj = TSNE(random_state=RS).fit_transform(X)

titanic_proj.shape
def scatter(x, colors):

    """this function plots the result

    - x is a two dimensional vector

    - colors is a code that tells how to color them: it corresponds to the target

    """

    

    # We choose a color palette with seaborn.

    palette = np.array(sns.color_palette("hls", 2))



    # We create a scatter plot.

    f = plt.figure(figsize=(10, 8))

    ax = plt.subplot(aspect='equal')

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,

                    c=palette[colors.astype(np.int)])

    

    ax.axis('off') # the axis will not be shown

    ax.axis('tight') # makes sure all data is shown

    

    # set title

    plt.title("Featurespace Visualization Titanic", fontsize=25)

    

    # legend with color patches

    survived_patch = mpatches.Patch(color=palette[1], label='Survived')

    died_patch = mpatches.Patch(color=palette[0], label='Died')

    plt.legend(handles=[survived_patch, died_patch], fontsize=20, loc=1)



    return f, ax, sc



# Use the data to draw ths scatter plot

scatter(titanic_proj, y)
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

print(roc_auc)