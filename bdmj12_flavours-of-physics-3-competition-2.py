### THIS CELL IS JUST THE EVALUATION PYTHON FILE 

import numpy
from sklearn.metrics import roc_curve, auc


def __rolling_window(data, window_size):
    """
    Rolling window: take window with definite size through the array

    :param data: array-like
    :param window_size: size
    :return: the sequence of windows

    Example: data = array(1, 2, 3, 4, 5, 6), window_size = 4
        Then this function return array(array(1, 2, 3, 4), array(2, 3, 4, 5), array(3, 4, 5, 6))
    """
    shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)
    strides = data.strides + (data.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def __cvm(subindices, total_events):
    """
    Compute Cramer-von Mises metric.
    Compared two distributions, where first is subset of second one.
    Assuming that second is ordered by ascending

    :param subindices: indices of events which will be associated with the first distribution
    :param total_events: count of events in the second distribution
    :return: cvm metric
    """
    target_distribution = numpy.arange(1, total_events + 1, dtype='float') / total_events
    subarray_distribution = numpy.cumsum(numpy.bincount(subindices, minlength=total_events), dtype='float')
    subarray_distribution /= 1.0 * subarray_distribution[-1]
    return numpy.mean((target_distribution - subarray_distribution) ** 2)


def compute_cvm(predictions, masses, n_neighbours=200, step=50):
    """
    Computing Cramer-von Mises (cvm) metric on background events: take average of cvms calculated for each mass bin.
    In each mass bin global prediction's cdf is compared to prediction's cdf in mass bin.

    :param predictions: array-like, predictions
    :param masses: array-like, in case of Kaggle tau23mu this is reconstructed mass
    :param n_neighbours: count of neighbours for event to define mass bin
    :param step: step through sorted mass-array to define next center of bin
    :return: average cvm value
    """
    predictions = numpy.array(predictions)
    masses = numpy.array(masses)
    assert len(predictions) == len(masses)

    # First, reorder by masses
    predictions = predictions[numpy.argsort(masses)]

    # Second, replace probabilities with order of probability among other events
    predictions = numpy.argsort(numpy.argsort(predictions, kind='mergesort'), kind='mergesort')

    # Now, each window forms a group, and we can compute contribution of each group to CvM
    cvms = []
    for window in __rolling_window(predictions, window_size=n_neighbours)[::step]:
        cvms.append(__cvm(subindices=window, total_events=len(predictions)))
    return numpy.mean(cvms)


def __roc_curve_splitted(data_zero, data_one, sample_weights_zero, sample_weights_one):
    """
    Compute roc curve

    :param data_zero: 0-labeled data
    :param data_one:  1-labeled data
    :param sample_weights_zero: weights for 0-labeled data
    :param sample_weights_one:  weights for 1-labeled data
    :return: roc curve
    """
    labels = [0] * len(data_zero) + [1] * len(data_one)
    weights = numpy.concatenate([sample_weights_zero, sample_weights_one])
    data_all = numpy.concatenate([data_zero, data_one])
    fpr, tpr, _ = roc_curve(labels, data_all, sample_weight=weights)
    return fpr, tpr


def compute_ks(data_prediction, mc_prediction, weights_data, weights_mc):
    """
    Compute Kolmogorov-Smirnov (ks) distance between real data predictions cdf and Monte Carlo one.

    :param data_prediction: array-like, real data predictions
    :param mc_prediction: array-like, Monte Carlo data predictions
    :param weights_data: array-like, real data weights
    :param weights_mc: array-like, Monte Carlo weights
    :return: ks value
    """
    assert len(data_prediction) == len(weights_data), 'Data length and weight one must be the same'
    assert len(mc_prediction) == len(weights_mc), 'Data length and weight one must be the same'

    data_prediction, mc_prediction = numpy.array(data_prediction), numpy.array(mc_prediction)
    weights_data, weights_mc = numpy.array(weights_data), numpy.array(weights_mc)

    assert numpy.all(data_prediction >= 0.) and numpy.all(data_prediction <= 1.), 'Data predictions are out of range [0, 1]'
    assert numpy.all(mc_prediction >= 0.) and numpy.all(mc_prediction <= 1.), 'MC predictions are out of range [0, 1]'

    weights_data /= numpy.sum(weights_data)
    weights_mc /= numpy.sum(weights_mc)

    fpr, tpr = __roc_curve_splitted(data_prediction, mc_prediction, weights_data, weights_mc)

    Dnm = numpy.max(numpy.abs(fpr - tpr))
    return Dnm


def roc_auc_truncated(labels, predictions, tpr_thresholds=(0.2, 0.4, 0.6, 0.8),
                      roc_weights=(4, 3, 2, 1, 0)):
    """
    Compute weighted area under ROC curve.

    :param labels: array-like, true labels
    :param predictions: array-like, predictions
    :param tpr_thresholds: array-like, true positive rate thresholds delimiting the ROC segments
    :param roc_weights: array-like, weights for true positive rate segments
    :return: weighted AUC
    """
    assert numpy.all(predictions >= 0.) and numpy.all(predictions <= 1.), 'Data predictions are out of range [0, 1]'
    assert len(tpr_thresholds) + 1 == len(roc_weights), 'Incompatible lengths of thresholds and weights'
    fpr, tpr, _ = roc_curve(labels, predictions)
    area = 0.
    tpr_thresholds = [0.] + list(tpr_thresholds) + [1.]
    for index in range(1, len(tpr_thresholds)):
        tpr_cut = numpy.minimum(tpr, tpr_thresholds[index])
        tpr_previous = numpy.minimum(tpr, tpr_thresholds[index - 1])
        area += roc_weights[index - 1] * (auc(fpr, tpr_cut, reorder=True) - auc(fpr, tpr_previous, reorder=True))
    tpr_thresholds = numpy.array(tpr_thresholds)
    # roc auc normalization to be 1 for an ideal classifier
    area /= numpy.sum((tpr_thresholds[1:] - tpr_thresholds[:-1]) * numpy.array(roc_weights))
    return area
def check_ag_test(model,var):
    check_agreement = pd.read_csv(folder + 'check_agreement.csv', index_col='id')
    agreement_probs = model.predict_proba(check_agreement[var])[:, 1]
    
    ks = compute_ks(
        agreement_probs[check_agreement['signal'].values == 0],
        agreement_probs[check_agreement['signal'].values == 1],
        check_agreement[check_agreement['signal'] == 0]['weight'].values,
        check_agreement[check_agreement['signal'] == 1]['weight'].values)
    print('KS metric', ks, ks < 0.09)
    return ks<0.09

def check_corr_test(model,var):

    check_correlation = pd.read_csv(folder + 'check_correlation.csv', index_col='id')
    correlation_probs = model.predict_proba(check_correlation[var])[:, 1]
    cvm = compute_cvm(correlation_probs, check_correlation['mass'])
    print('CvM metric', cvm, cvm < 0.002)
    return cvm<0.002

def comp_auc(model,var):
    train_eval = train[train['min_ANNmuon'] > 0.4]
    train_probs = model.predict_proba(train_eval[var])[:, 1]
    AUC = roc_auc_truncated(train_eval['signal'], train_probs)
    print('AUC', AUC)
    return AUC

## combine tests into one function

def eval(model,var):
    check_ag_test(model,var)
    check_corr_test(model,var)
    comp_auc(model,var)
def pred_file(model,var):

    test = pd.read_csv(folder + 'test.csv', index_col='id')
    result = pd.DataFrame({'id': test.index})
    result['prediction'] = model.predict_proba(test[var])[:, 1]
    result.to_csv('prediction %s .csv' % version, index=False, sep=',')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

#import evaluation
#import evalfunctions

version = "1.0"
folder = '../input/'
train = pd.read_csv(folder + 'training.csv', index_col='id')
train.head()
plt.figure(figsize=(5,20))
sns.heatmap(train.corr()["signal"].to_frame().sort_values(by="signal", ascending=False), annot=True, center=0)
# this function returns a list of variables with a magnitude correlation with signal greater than n
# excluding some variables not to be included in the training

def significantFactors(n):
    x =[]
    corr = train.corr()["signal"]
    for i in range(len(corr)):
        if abs(corr[i])>n:
            if(corr.index[i] not in ["production", "min_ANNmuon","signal","mass"]):
                x.append(corr.index[i])
    return x
# these are some variables that we would definitely like to include 
##(mainly based on common sense and trial and error)

variables = train.drop(["production", "min_ANNmuon","signal","mass", # these are not to be included
                        "SPDhits", # including this makes agreement test fail
                        "FlightDistanceError" # this seems to worsen score - perhaps not relevant (noise)
                       ],axis=1).columns
variables
# ↓↓↓ I'm not sure this improves efficiency

#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#scaled_data=train
#scaled_data[variables] = scaler.fit_transform(scaled_data[variables])
#train=scaled_data
candidate_models = {}   # we'll store candidate models here

def test_model(model):
    #if the model passes the tests...
    if(check_corr_test(model,variables) and check_ag_test(model,variables)):
        #...add it to the candidates
        candidate_models[svc] = comp_auc(model,variables)
        print('passed')
    else:
        print('failed')
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=40, learning_rate=0.01, subsample=0.7,
                                      min_samples_leaf=10, max_depth=7, random_state=11)
gbc.fit(train[variables], train['signal'])

#eval(gbc,variables)

if(check_corr_test(gbc,variables) and check_ag_test(gbc,variables)):
    candidate_models[gbc] = comp_auc(gbc,variables)
    print('passed')
else:
    print("failed")
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train[variables], train['signal'])

#eval(lr,variables)

if(check_corr_test(lr,variables) and check_ag_test(lr,variables)):
    candidate_models[lr] = comp_auc(lr,variables)
    print('passed')
else:
    print("failed")
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(train[variables], train['signal'])

#eval(gnb,variables)

if(check_corr_test(gnb,variables) and check_ag_test(gnb,variables)):
    candidate_models[gnb] = comp_auc(gnb,variables)
    print('passed')
else:
    print("failed")
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier()

knc.fit(train[variables], train['signal'])

#eval(gbc,variables)

if(check_corr_test(knc,variables) and check_ag_test(knc,variables)):
    candidate_models[knc] = comp_auc(knc,variables)
    print('passed')
else:
    print("failed")
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth = 10,max_features=5 )
dtc.fit(train[variables], train['signal'])

#eval(dtc,variables)

if(check_corr_test(dtc,variables) and check_ag_test(dtc,variables)):
    candidate_models[dtc] = comp_auc(dtc,variables)
    print('passed')
else:
    print("failed")
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_depth = 15,max_features=5) # <<<< this was just from trial-and-error tweaking

rfc.fit(train[variables], train['signal'])

#eval(rfc,variables)

if(check_corr_test(rfc,variables) and check_ag_test(rfc,variables)):
    candidate_models[rfc] = comp_auc(rfc,variables)
    print('passed')
else:
    print("failed")
candidate_models
best_model = max(candidate_models, key=candidate_models.get)
type(best_model)
candidate_models[best_model]
pred_file(best_model,variables)