import numpy as np

import pandas as pd

import seaborn as sns

import itertools

import os

import matplotlib.pyplot as plt
# read in input data

train = pd.read_csv('../input/mnist_train.csv', dtype=int)

X_train = train.drop('label', axis=1)

y_train = train['label']



test = pd.read_csv('../input/mnist_test.csv', dtype=int)

X_test = test.drop('label', axis=1)

y_test = test['label']
# CONSTS

RANDOM_STATE = 1313361



# mapping consts

CLASSIFIER = 'classifier'

OUT_OF_BAG = 'oob_score'

DEPTH = 'depth'

TEST_PRED = 'test_pred'

PRED_PROB = 'prediction_probability'

DEC_FUNC = 'decision_function'



NAME = 'name'



depth_list = [10, 20, 30, 40, 50, 60]
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score



def find_best_max_depth(depth_list, classifier_generator, X_train, X_test, y_train, y_test):

    '''

    Runs through each depth in depth_list, creates a new classifier using

    classifier_generator. classifier_generator should be a function which

    takes as input a depth and returns a classifier. It should also set

    any parameters needed to generate an oob_score.

    i.e. bootstrap &/or oob_score.

    '''

    top_classifier = {

        CLASSIFIER: None,

        OUT_OF_BAG: -1,

        DEPTH: -1,

        TEST_PRED: None,

        DEC_FUNC: None,

        PRED_PROB: None,

    }

    for depth in depth_list:

        CLF = classifier_generator(depth=depth)

        CLF.fit(X_train, y_train)

        oob = CLF.oob_score_



        if oob > top_classifier[OUT_OF_BAG]:

            top_classifier[OUT_OF_BAG] = oob

            top_classifier[CLASSIFIER] = CLF

            top_classifier[DEPTH] = depth



    assert top_classifier[CLASSIFIER] is not None

    assert top_classifier[OUT_OF_BAG] != -1

    assert top_classifier[DEPTH] != -1



    # predict, get accuracy, decision function and prediction probability

    y_pred = top_classifier[CLASSIFIER].predict(X_test)

    top_classifier[TEST_PRED] = accuracy_score(y_test, y_pred)

    top_classifier[DEC_FUNC] = top_classifier[CLASSIFIER].oob_decision_function_

    top_classifier[PRED_PROB] = top_classifier[CLASSIFIER].predict_proba(X_test)



    print("Best model parameter: %d, oob_score: %f, actual prediction: %f" % (

        top_classifier[DEPTH], top_classifier[OUT_OF_BAG], top_classifier[TEST_PRED]

    ))

    return top_classifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier



def single_extra_tree(depth=None, seed=RANDOM_STATE):

    '''Creates a single ExtraTreesClassifier with only one tree.'''

    return ExtraTreesClassifier(

        max_depth=depth, n_estimators=1,

        bootstrap=True, random_state=seed,

        oob_score=True,

    )
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline



# create bagged pca first, otherwise kernel runs out of memory

best_ada_depth = 20 # best_bagged_clf[DEPTH] # 

def create_bagged_pca_pipeline(depth=None, seed=RANDOM_STATE, best_depth=best_ada_depth):

    '''

    depth is actually used as num_components.

    '''

    pca = PCA(n_components=depth, svd_solver='randomized', random_state=seed)

    pipe = Pipeline(

        [('pca', pca),

        ('ada_xt',

            AdaBoostClassifier(

                base_estimator=single_extra_tree(depth=best_depth),

                n_estimators=10,

                random_state=seed))

        ])

    return BaggingClassifier(base_estimator=pipe, n_estimators=30, bootstrap=True, oob_score=True, n_jobs=-1, random_state=seed)



num_components_list = [20, 40, 60]

# 20

best_pipeline_clf = find_best_max_depth(

    num_components_list, create_bagged_pca_pipeline,

    X_train, X_test, y_train, y_test

)
def create_extra_trees(depth=None, seed=RANDOM_STATE):

    '''Creates an ExtraTreesClassifier'''

    return ExtraTreesClassifier(

        n_estimators=300,

        max_depth=depth,

        n_jobs=-1,

        bootstrap=True,

        oob_score=True,

        random_state=seed,

    )



# 50

best_etc = find_best_max_depth(

    depth_list, create_extra_trees,

    X_train, X_test, y_train, y_test

)
def create_bagging_predictor(depth=None, seed=RANDOM_STATE):

    boost = AdaBoostClassifier(base_estimator=single_extra_tree(depth=depth), n_estimators=10, random_state=seed)

    return BaggingClassifier(base_estimator=boost, n_estimators=30, bootstrap=True, oob_score=True, n_jobs=-1, random_state=seed)



# 20

best_bagged_clf = find_best_max_depth(

    depth_list, create_bagging_predictor,

    X_train, X_test, y_train, y_test

)
# check overall prediction

best_etc[NAME] = 'extra_trees'

best_bagged_clf[NAME] = 'ada_bagged'

best_pipeline_clf[NAME] = 'ada_bagged_pipeline'



trained_models = [best_etc, best_bagged_clf, best_pipeline_clf]

pred_prob_list = [clf[DEC_FUNC] for clf in trained_models]

clf_names = [clf[NAME] for clf in trained_models]



def average_oob_scores(pred_prob_list):

    '''Averages a list of oob_decision_function_.'''

    return sum(pred_prob_list) / len(pred_prob_list)



def get_best_model_pp_avg(pred_prob_list, subset_names, y_train):

    '''

    Goes through every subset of a list consisting of

    oob_decision_function_ values. Uses subset_names

    to name each entry to the list of oob_decision_function_.

    Prints the concatted models accuracy using y_train.

    '''

    if not pred_prob_list:

        return None

    if len(pred_prob_list) != len(subset_names):

        raise RuntimeError(

            'Pred probailities needs to have same amount of entries as subset names'

        )

    best_subset_score = -1

    best_subset = None

    # for every possible subset combination

    for num_entries in range(1, len(pred_prob_list) + 1):

        subsets = itertools.combinations(pred_prob_list, num_entries)

        named_subsets = itertools.combinations(subset_names, num_entries)



        # for every subset, check how it should perform

        for clf_subset, subset in zip(named_subsets, subsets):

            avg_decsion_funcs = average_oob_scores(subset)

            avg_predictions = np.argmax(avg_decsion_funcs, axis=1)

            accuracy = accuracy_score(y_train, avg_predictions)

            print(clf_subset, accuracy)



best_model = get_best_model_pp_avg(pred_prob_list, clf_names, y_train)
from sklearn.ensemble import VotingClassifier



# not actually the best because the best was only a single model.

# I wanted to run voting with more than one model

best_subset = [best_etc, best_bagged_clf]

estimators = [(clf_entries[NAME], clf_entries[CLASSIFIER]) for clf_entries in best_subset]



voting_clf = VotingClassifier(

    estimators=estimators,

    voting='soft'

)

voting_clf.fit(X_train, y_train)

voted_y_pred = voting_clf.predict(X_test)

print('extra_trees + ada_bagged acheived %.2f%% accuracy' % (accuracy_score(y_test, voted_y_pred) * 100))


from sklearn.linear_model import LogisticRegression



# get the x meta train set using the decision function from every model

X_train_meta = np.concatenate([clf[DEC_FUNC] for clf in trained_models], axis=1)



# get the x meta test set using the prediction probability from every model

X_test_meta = np.concatenate([clf[PRED_PROB] for clf in trained_models], axis=1)



# train and predict a logistic regressor with the meta sets and y train/test data

log_regressor = LogisticRegression(C=50, random_state=RANDOM_STATE)

log_regressor.fit(X_train_meta, y_train)

log_reg_pred = log_regressor.predict(X_test_meta)



# print the results

print('logistic regressor with meta sets acheived %.2f%% accuracy' % (accuracy_score(y_test, log_reg_pred) * 100))

def get_digit_missclassifications(y_test, y_pred):

    '''

    '''

    digit_range = range(0, 10)

    # [[], [], [], [], [], [], [], [], []]

    digit_mapping = [[0 for digit in digit_range] for _ in digit_range]

    for expected, prediction in  zip(y_test, y_pred):

        digit_mapping[expected][prediction] += 1



    def make_missclassified_dict(missclassifications):

        return { digit: missclassification for digit, missclassification in enumerate(missclassifications) }



    digit_dict = {}

    for digit, missclassifications in enumerate(digit_mapping):

        digit_dict[digit] = make_missclassified_dict(missclassifications)

    return digit_dict



def plot_digit(X_test, digit_index, missclassifications, img_title, bar_title):

    '''

    Prints the digit at digit_index in X_test

    with the title set.

    '''

    pixels = pd.DataFrame(X_test.iloc[digit_index,:])

    pixels = pixels.values.reshape((28,28))

    # plt.subplot(1, 2, 1)

    plt.suptitle(img_title)

    plt.imshow(pixels, cmap='gray')

    plt.show()



    mclf_x_labels = missclassifications.keys()

    mclf_counts = missclassifications.values()

    # plt.subplot(1, 2, 2)

    plt.suptitle(bar_title)

    plt.bar(mclf_x_labels, mclf_counts, align='center', alpha=1)



    plt.xlabel('Digit')

    plt.ylabel('Misclassification counts')

    plt.show()
# finds missclassified predictions

def plot_missclassifications(y_test, y_pred, X_test):

    '''

    Goes through y_test and y_pred and plots

    one digit from each digit class.

    '''

    missclassifications = get_digit_missclassifications(y_test, log_reg_pred)

    found = []

    missclassified_as = []



    # go through the expected predictions

    idx = -1

    for expected, prediction in zip(y_test, y_pred):

        idx += 1



        # if prediction was valid or we've already found a missclassifcation

        if (expected == prediction

            or expected in set(found)

            or prediction in set(missclassified_as)):

            continue

        



        found.append(expected)

        missclassified_as.append(prediction)



        # print the missclassification information

        title = '%d missclassified as %d ' % (y_test[idx], y_pred[idx])

        bar_title = 'missclassification counts for "%d"' % expected



        # setting axis labels sucks

        missclassifications[expected][expected] = -1

        plot_digit(X_test, idx, missclassifications[expected], title, bar_title)



plot_missclassifications(y_test, log_reg_pred, X_test)
CORRECT = 'correct'

TOTAL = 'total'



def get_digit_prediction_accuracies(y_test, y_pred):

    '''

    Gets the prediction accuracy for each digit in the dataset

    '''

    digit_range = range(0, 10)

    digit_mapping = [{ CORRECT: 0, TOTAL: 0 } for _ in digit_range]

    for actual, prediction in zip(y_test, y_pred):

        assert actual in set(digit_range)

        if actual == prediction:

            digit_mapping[actual][CORRECT] += 1

        digit_mapping[actual][TOTAL] += 1



    def get_pred(predictions):

        '''Gets the prediction accuracy.'''

        return (predictions[CORRECT] / predictions[TOTAL]) # * 100



    return [get_pred(predictions) for predictions in digit_mapping]      



digit_accuracies = get_digit_prediction_accuracies(y_test, log_reg_pred)



# print the accuracy

for digit, accuracy in enumerate(digit_accuracies):

    print('%d: %.2f%% accuracy' % (digit, accuracy))

# def plot_digit_prediction_accuracies(digit_accuracies):

#     '''

#     Plots digit prediction accuracy using the output

#     from get_digit_prediction_accuracies.

#     '''

#     # generate 0, 1, ... 9

#     digits = [str(digit) for digit in range(0, len(digit_accuracies))]



#     # plot the figure

#     plt.figure(figsize=(10, 7))

#     plt.bar(digits, digit_accuracies, align='center', alpha=1)

#     plt.title('Digit Prediction Accuracies')

#     plt.xlabel('Digit')

#     plt.ylabel('Model Prediction Accuracy')

#     plt.show()



# plot_digit_prediction_accuracies(digit_accuracies)