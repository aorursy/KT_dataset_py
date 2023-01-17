import numpy as np

import pandas as pd

import seaborn as sns

import os

import matplotlib.pyplot as plt
# CONSTS USED

# pinned random state

RANDOM_STATE = 1313361



RFC = 'RandomForestClassifier'

ETC = 'ExtraTreesClassifier'


train = pd.read_csv('../input/fashion-mnist_train.csv', dtype=int)

X_train = train.drop('label', axis=1)

y_train = train['label']

test = pd.read_csv('../input/fashion-mnist_test.csv', dtype=int)

X_test = test.drop('label', axis=1)

y_test = test['label']
def plot_matrix(matrix, title='', use_sns=True):

    '''

    Creates a heatmap of the input matrix.



    Arguments:

        matrix: The matrix to plot. Must have a width and

            a height defined when matrix.shape is called.

        title: If provided, heatmap title will be set to

            this value.

        use_sns: If True, heatmap will be created

            using seaborn. If False heatmap will be created

            using matplotlib. Default True.

    '''

    fig = plt.figure(figsize=matrix.shape)



    if title:

        plt.suptitle(title)



    # use matplotlib instead

    if not use_sns:

        ax = fig.add_subplot(111)

        cax = ax.matshow(matrix)

        fig.colorbar(cax)

        return



    sns.heatmap(matrix, annot=True)



X_matrix = X_train[0:1].values.reshape(28,28)

plot_matrix(X_matrix, title='GaussianNB Matrix Heatmap')
# get an initial benchmark on how "hard" the dataset is

# fit and redict using GaussianNB

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



def test_classifier(CLF, X_train, X_test, y_train):

    '''Fits and predicts using input params'''

    classifier = CLF()

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    return classifier, predictions



naive_bayes, y_pred = test_classifier(GaussianNB, X_train, X_test, y_train)

model_accuracy = accuracy_score(y_test, y_pred)

print('Benchmark model accuracy: %.2f%%' % (model_accuracy * 100))
import itertools



def get_ft_depth_pairs():

    '''

    Creates a list with all of the pairings of feature_values

    and depth_values

    '''

    feature_values = [1, 4, 16, 64,'auto']

    depth_values = [1, 4, 16, 64, None]

    pairing = [

        feature_values,

        depth_values

    ]

    return list(itertools.product(*pairing))



print(get_ft_depth_pairs())
from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

import time



def grid_search(CLF, X_train, y_train, n_estimators=30):

    '''

    Uses all possible pairs from get_ft_depth_pairs

    to find the best hyper params for the input

    classifier.

    '''

    possible_pairs = get_ft_depth_pairs()



    for pairing in possible_pairs:

        features_paramter = pairing[0]

        max_depth_parameter = pairing[1]



        # bootstrap=true to get oob_score_ for ExtraTrees

        # oob_score=true to get oob_score_ for RandomForest

        classifier = CLF(

            max_features=features_paramter,

            max_depth=max_depth_parameter,

            n_estimators=n_estimators,

            random_state=RANDOM_STATE,

            oob_score=True,

            bootstrap=True

        )



        start = time.time()

        classifier.fit(X_train, y_train)

        end = time.time()

        oob_score = classifier.oob_score_



        print('max_features=%s max_depth=%s oob_score=%f execution_time=%.2fs' % (

            features_paramter, max_depth_parameter, oob_score, end-start)

        )



classifiers = [RandomForestClassifier, ExtraTreesClassifier]

for classifier in classifiers:

    print('running %s for best hyper_params' % str(classifier))

    grid_search(classifier, X_train, y_train)

    

        


from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier



# best models, trained

optimal_rfc = RandomForestClassifier(

    max_features=4, max_depth=None, random_state=RANDOM_STATE, n_estimators=30

)

optimal_rfc.fit(X_train, y_train)



optimal_etc = ExtraTreesClassifier(

    max_features=16, max_depth=64, random_state=RANDOM_STATE, n_estimators=100

)

optimal_etc.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix



def get_results(trained_clf, X_test, y_test):

    '''Predicts using X_test, prints accuracy and confusion matrix.'''

    model_pred = trained_clf.predict(X_test)

    model_acc = accuracy_score(y_test, model_pred)

    cm = confusion_matrix(y_test, model_pred)



    print('model acheived %.2f%% accuracy\ncr: %s' % (

        (model_acc * 100), cm)

    )

    return model_pred, cm



print('getting accuracy and cm for ' + RFC)

rfc_pred, rfc_cm = get_results(optimal_rfc, X_test, y_test)



print()



print('getting accuracy and cm for ' + ETC)

etc_pred, etc_cm = get_results(optimal_etc, X_test, y_test)
FIG_SIZE = (10, 7)

ANNOT = True

matricies = {

    RFC: rfc_cm,

    ETC: etc_cm,

}



for model_name in matricies:

    cm = matricies[model_name]

    plt.figure(figsize=FIG_SIZE)

    plt.suptitle(model_name)

    sns.heatmap(cm, annot=ANNOT)

column_names = list(X_train)



rfc_ft_impt = optimal_rfc.feature_importances_.reshape(28, 28)

etc_ft_impt = optimal_etc.feature_importances_.reshape(28, 28)



plot_matrix(rfc_ft_impt, title='RandomForestClassifier Feature Importance')
plot_matrix(etc_ft_impt, title='ExtraTreesClassifier Feature Importance')