import numpy as np

import pandas as pd

import seaborn as sns

import os
# CONSTS USED

# pinned random state

RANDOM_STATE=1313361



# model keys

SGD = 'sgd_classifier'

RFOREST = 'random_forest_30_branches'

NAIVE_BAYES = 'naive_bayes'



# dict storing keys

CV = 'cv'

CV_SCORE = 'cv_score'

FIT = 'fit'

FPR = 'fpr'

TPR = 'tpr'

CLASSIFIED_BY = 'classified_by'
from sklearn.base import clone

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import roc_curve

import warnings



with warnings.catch_warnings():

    warnings.filterwarnings("ignore",category=DeprecationWarning)



def get_pred_results(y_pred, y_actual):

    '''Gets the prediction accuracy'''

    n_correct = sum(y_pred == y_actual)

    return n_correct / len(y_actual)



def benchmark_predictions(x_dat, y_dat, model):

    '''computes the cross_val and prediction score for a model given x and y dat'''

    warnings.filterwarnings("ignore")

    x_train, x_test, y_train, y_test = train_test_split(x_dat, y_dat, test_size=0.2, stratify=y_dat, random_state=RANDOM_STATE)

    y_train, y_test = y_train == 1.0, y_test == 1.0



    # get predictions via cross-validation

    cv_pred = cross_val_predict(model, x_dat, y_dat, cv=10) == 1.0

    cv_result = get_pred_results(cv_pred, y_dat)

    cv_fpr, cv_tpr, _ = roc_curve(y_dat, cv_pred)

    

    # get predictions via just using the inital split

    cloned_model = clone(model)

    cloned_model.fit(x_train, y_train)

    y_pred = cloned_model.predict(x_test) == 1.0

    fit_result = get_pred_results(y_pred, y_test)



    # figure out which was more accurate

    max_val = cv_result if cv_result >= fit_result else fit_result

    classifier = CV if cv_result >= fit_result else FIT

    return {

        CV: cv_result,

        CV_SCORE: np.mean(cross_val_score(model, x_dat, y_dat, cv=10, scoring='accuracy')),

        FIT: fit_result,

        FPR: cv_fpr,

        TPR: cv_tpr,

        'max': max_val,

        CLASSIFIED_BY: classifier

    }

import itertools 



def generate_subsets(full_set):

    '''Generates every possible subset for `full_set`'''

    set_length = len(full_set)

    # can't generate any subsets

    assert set_length >= 2



    # inclusive subset range

    set_length += 1

    subsets = []

    for i in range(2, set_length):

        subsets += list(itertools.combinations(full_set, i))

    return subsets
from sklearn.model_selection import cross_val_score



def predict_all_subsets(full_data, y_dat, subsets, model):

    '''

    Goes through every possible subset and finds gets

    the cross_validation and prediction accuracy scores.

    Returns the information in a dict

    '''

    results = {}

    for index, subset in enumerate(subsets):

        x_dat = full_data.iloc[:, list(subset)]

        results[subset] = benchmark_predictions(x_dat, y_dat, model)

    return results
from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB



def get_model_list(random_seed):

    '''Don't get models until actually requests'''

    return {

        SGD: SGDClassifier(random_state=random_seed), 

        RFOREST: RandomForestClassifier(n_estimators=30, random_state=random_seed),

        NAIVE_BAYES: GaussianNB()

    }

input_csv_loc = '../input/wisconsin_breast_cancer.csv'

bccf = pd.read_csv(input_csv_loc)



bccf_clean = bccf.dropna()

bccf_clean.describe()
# split the data

from sklearn.model_selection import train_test_split



x_indexes = [1,2,3,4,5,6,7,8]

# generates all possible subsets, uses indexes for easier looping

subsets = generate_subsets(x_indexes)



x_dat = bccf_clean.iloc[:, x_indexes[0]:x_indexes[-1]+1]



# select all entries, then get col 10

y_dat = bccf_clean.iloc[:, 10]

y_dat_bin = y_dat == 1.0
# get the list of models that are being tested

models = get_model_list(RANDOM_STATE)

results = {}

for model in models:

    print('calculating subset accuracy for %s' % model)

    # for each model get the accuracy for every possible subset

    results[model] = predict_all_subsets(bccf_clean, y_dat_bin, subsets, models[model])

print('found subsets')
def look_up_subset(data, subset):

    '''Gets the column names of a subset'''

    columns = []

    for index in subset:

        columns.append(data.columns[index])

    return columns



def get_perfect_cv_scores(results, num_to_get):

    '''Gets the `highest_count` number of highest cross_val_score subsets'''

    model_ss = {}

    if num_to_get <= 0:

        return None

    for model in results:

        model_results = results[model]

        cv_accuracies_lambda = lambda ss : model_results[ss][CV_SCORE]

        cv_accuracies = []

        cv_ss_scores = {}

        for subset in model_results:

            cv_score = cv_accuracies_lambda(subset)

            cv_accuracies.append(cv_score)

            cv_ss_scores[cv_score] = subset

        cv_accuracies = sorted(cv_accuracies, key=float, reverse=True)

        model_best = {}

        for _, accuracy in zip(range(0, num_to_get), cv_accuracies):

            subset = cv_ss_scores[accuracy]

            model_best[subset] = {

                CV_SCORE: accuracy,

                CV: model_results[subset][CV],

                FIT: model_results[subset][FIT]

            }

        model_ss[model] = model_best

    return model_ss



best_cv_scores = get_perfect_cv_scores(results, 1)

for model in best_cv_scores:

    print(model)

    scores = best_cv_scores[model]

    for subset in scores:

        col_names = look_up_subset(bccf_clean, subset)

        print('\tsubset_indexes: %s\n\tsubset: %s\n\tcv_score average: %f\n\tcv_prediction: %f\n\tfit_prediction: %f\n' % (

            subset, col_names, scores[subset][CV_SCORE], scores[subset][CV], scores[subset][FIT])

         )

    print()
def print_max_details(data, results):

    '''Prints the maximum accuracy (i.e. fit or cross val) scored by a model'''

    for model_key in results:

        model_results = results[model_key]

        max_subset = None

        max_val = 0.0

        for subset in model_results:

            subset_results = model_results[subset]

            ss_max_val = subset_results['max']

            if ss_max_val > max_val:

                max_val = ss_max_val

                max_subset = subset

        print('model: %s, max_subset: %s (%s) accuracy: %s' % (model_key, max_subset, subset_results[CLASSIFIED_BY], max_val))

        print('subset column names: ' + ', '.join(look_up_subset(data, max_subset)))

        print()





def print_results(data, results):

    '''Prints the results scored by every subset of every model in results'''

    for model_key in results:

        model_results = results[model_key]

        for subset in model_results:

            result = model_results[subset]

            # prediction, trained_model keys

            fit = result[FIT]

            cv_avg = result[CV]

            difference = cv_avg - fit

            print('%s: subset: %s (%s), fit: %s, cv: %s, dif: %s' % (model_key, subset, result[CLASSIFIED_BY], fit, cv_avg, difference))

            print('col names: ' + ', '.join(look_up_subset(data, subset)))

            print()



def print_results_for_subset(data, results, subset):

    '''Prints every models results for a specific subset'''

    for model_key in results:

        model_results = results[model_key]

        if subset not in model_results:

            print('could not find %s in %s' % (subset, model_key))

            return

        result = model_results[subset]

        # prediction, trained_model keys

        fit = result[FIT]

        cv_avg = result[CV]

        difference = cv_avg - fit

        subset_col_names = look_up_subset(data, subset)

        print('%s: subset: %s (%s), fit: %s, cv: %s, dif: %s' % (model_key, subset, result[CLASSIFIED_BY], fit, cv_avg, difference))

        print('col names: ' + ', '.join(look_up_subset(data, subset)))

        print()



print_max_details(bccf_clean, results)
def extract_scatter_plot_data(model_results):

    '''

    Helper function for produce_scatter_plot.

    Extracts the cv and test-accuracy data

    from the results dict.

    '''

    cv_results = []

    fit_results = []

    for subset in model_results:

        result = model_results[subset]

        # prediction, trained_model keys

        cv_results.append(result[CV])

        fit_results.append(result[FIT])

    return (cv_results, fit_results)



def produce_scatter_plot(results, results_key):

    '''creates a scatter plot for an entry in results'''

    assert results_key in results

    sp_data = extract_scatter_plot_data(results[results_key])

    sp = sns.scatterplot(x=sp_data[0], y=sp_data[1])

    sp.set(title=results_key, xlabel='cv-accuracy', ylabel='test-accuracy')
# sgd scatter plot

produce_scatter_plot(results, SGD)
# randomforest scatter plot

produce_scatter_plot(results, RFOREST)
# naive bayes scatter plot

produce_scatter_plot(results, NAIVE_BAYES)
# roc curve making

def find_highest_avg_subset(results):

    '''

    Finds the subset that has the highest average

    accross all of the models in results.

    '''

    subset_results = [0] * len(subsets)

    for model_key in results:

        model_results = results[model_key]

        for index, subset in enumerate(model_results):

            subset_results[index] += model_results[subset][CV]



    # average the array by number of models in results

    subset_sums = [subset_sum / len(results) for subset_sum in subset_results]

    max_subset = max(subset_sums)



    # get the index where the max value occured and use it

    # to get the corresponding subset

    index_of_ss = subset_sums.index(max_subset)

    return subsets[index_of_ss]



highest_avg_subset = find_highest_avg_subset(results)

print(look_up_subset(bccf_clean, highest_avg_subset))

print_results_for_subset(results, highest_avg_subset)
import matplotlib.pyplot as plt



def create_roc_graph(results, subset):

    '''Creates a roc curve graph with every models curve on it'''

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot([0, 1], [0, 1], 'k--')

    for model_key in results:

        model_results = results[model_key]

        specific_ss = model_results.get(subset,)

        ax.plot(specific_ss[FPR], specific_ss[TPR], linewidth=2, label=model_key)

    plt.legend(loc='lower right')

    plt.xlabel(FPR)

    plt.ylabel(TPR)

    plt.suptitle('roc curve for all models')

    plt.show()



create_roc_graph(results, highest_avg_subset)
# for all selected data cols

create_roc_graph(results, subsets[-1])