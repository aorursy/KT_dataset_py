import numpy as np
import pandas as pd
import lightgbm as lgb
import os

from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from hep_ml.gradientboosting import UGradientBoostingClassifier
from hep_ml.losses import BinFlatnessLossFunction
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

def feature_importance(forest, X_train):
    ranked_list = []
    
    importances = forest.feature_importances_

    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]) + " - " + X_train.columns[indices[f]])
        ranked_list.append(X_train.columns[indices[f]])
    
    return ranked_list
data_path = "../input"
train = pd.read_csv(os.path.join(data_path, 'training.csv'), index_col='id')
test = pd.read_csv(os.path.join(data_path, 'test.csv'), index_col='id')
check_agreement = pd.read_csv(os.path.join(data_path, 'check_agreement.csv'), index_col='id')

trainids = train.index.values
testids = test.index.values
caids = check_agreement.index.values
trainsignals = train.signal.ravel()
signal = train.signal
def add_features(df):
    # features used by the others on Kaggle
    df['NEW_FD_SUMP']=df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])
    df['NEW5_lt']=df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    #df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError'] # modified to:
    df['flight_dist_sig2'] = (df['FlightDistance']/df['FlightDistanceError'])**2
    # features from phunter
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    df['NEW_IP_dira'] = df['IP']*df['dira']
    df['p0p2_ip_ratio']=df['IP']/df['IP_p0p2']
    df['p1p2_ip_ratio']=df['IP']/df['IP_p1p2']
    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)
    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
    df['iso_min'] = df.loc[:, ['isolationa', 'isolationb', 'isolationc','isolationd', 'isolatione', 'isolationf']].min(axis=1)
    # My:
    # new combined features just to minimize their number;
    # their physical sense doesn't matter
    df['NEW_iso_abc'] = df['isolationa']*df['isolationb']*df['isolationc']
    df['NEW_iso_def'] = df['isolationd']*df['isolatione']*df['isolationf']
    df['NEW_pN_IP'] = df['p0_IP']+df['p1_IP']+df['p2_IP']
    df['NEW_pN_p']  = df['p0_p']+df['p1_p']+df['p2_p']
    df['NEW_IP_pNpN'] = df['IP_p0p2']*df['IP_p1p2']
    df['NEW_pN_IPSig'] = df['p0_IPSig']+df['p1_IPSig']+df['p2_IPSig']
    #My:
    # "super" feature changing the result from 0.988641 to 0.991099
    df['NEW_FD_LT']=df['FlightDistance']/df['LifeTime']
    return df
train = add_features(train)
test = add_features(test)
check_agreement = add_features(check_agreement)
p1 = 11.05855369567871094
p2 = 0.318310
p3 = 1.570796

def Output(p):
    return 1/(1.+np.exp(-p))

def GP(data):
    return Output(  1.0*np.tanh(((((((((data["IPSig"]) + (data["ISO_SumBDT"]))) - (np.minimum(((-2.0)), ((data["ISO_SumBDT"])))))) / (data["ISO_SumBDT"]))) / (np.minimum((((-1.0*((data["ISO_SumBDT"]))))), ((data["IPSig"])))))) +
                    1.0*np.tanh((-1.0*((((data["iso"]) + (((((((((((((data["VertexChi2"]) + ((3.0)))) / (data["ISO_SumBDT"]))) * (data["IP"]))) * 2.0)) / (data["ISO_SumBDT"]))) * (((((((((((data["VertexChi2"]) + ((3.0)))) / (data["ISO_SumBDT"]))) * (data["IP"]))) * 2.0)) / (data["ISO_SumBDT"])))))))))) +
                    1.0*np.tanh((-1.0*(((((((((data["IPSig"]) * ((((data["iso"]) + (((data["IP"]) * 2.0)))/2.0)))) + (np.tanh((data["p0_IsoBDT"]))))/2.0)) * ((((data["p0_IsoBDT"]) + (data["IPSig"]))/2.0))))))) +
                    1.0*np.tanh(((np.minimum(((np.cos((((np.cos((((data["p0_track_Chi2Dof"]) * (np.cos((data["p0_track_Chi2Dof"]))))))) * (np.log((data["IP_p0p2"])))))))), ((np.cos((data["p0_track_Chi2Dof"])))))) * (data["p0_track_Chi2Dof"]))) +
                    1.0*np.tanh((((((((((p1)) / (((((p1)) + (((((data["SPDhits"]) / 2.0)) / 2.0)))/2.0)))) - (data["IP"]))) - (((data["SPDhits"]) / (data["p1_pt"]))))) * 2.0)) +
                    1.0*np.tanh((((((((((((((data["CDF3"]) / (data["dira"]))) > (data["CDF3"]))*1.)) > (data["CDF3"]))*1.)) / 2.0)) + ((-1.0*((((((data["CDF3"]) * (data["p2_track_Chi2Dof"]))) * (((data["CDF3"]) * (data["p2_track_Chi2Dof"])))))))))/2.0)) +
                    1.0*np.tanh((((-1.0*((((data["DOCAthree"]) / (data["CDF2"])))))) + (np.minimum(((((data["p2_pt"]) / (data["p0_p"])))), ((np.minimum(((data["CDF2"])), ((((np.sin((p3))) / 2.0)))))))))) +
                    1.0*np.tanh(np.minimum((((-1.0*(((((((data["FlightDistance"]) < (data["IPSig"]))*1.)) / 2.0)))))), ((((np.minimum(((np.cos((np.log((data["p0_pt"])))))), ((np.cos((data["p1_track_Chi2Dof"])))))) / (p2)))))) +
                    1.0*np.tanh(((np.sin((np.where(data["iso"]>0, ((((data["iso"]) - ((-1.0*((((data["IPSig"]) / 2.0))))))) / 2.0), ((((3.0) * (data["IP"]))) * 2.0) )))) / 2.0)) +
                    1.0*np.tanh(((((np.cos(((((data["ISO_SumBDT"]) + (p2))/2.0)))) - (np.sin((np.log((data["p1_eta"]))))))) - ((((((data["ISO_SumBDT"]) + (np.cos((data["p2_IsoBDT"]))))/2.0)) * ((((data["ISO_SumBDT"]) + (np.cos((data["p2_IsoBDT"]))))/2.0)))))))
tr_preds_1 = GP(train).values
test_preds_1 = GP(test).values
ca_preds_1 = GP(check_agreement).values

test_predictions = pd.DataFrame({'id':testids,'predictions_1':test_preds_1})
train_predictions_all = pd.DataFrame({'id':trainids,'predictions_1':tr_preds_1})
ca_predictions = pd.DataFrame({'id':caids,'predictions_1':ca_preds_1})
# since the target is not used for this model we can add the feature to our data without any leakage
train['lines'] = tr_preds_1
check_agreement['lines'] = ca_preds_1
test['lines'] = test_preds_1
agreement_probs = ca_predictions.predictions_1

ks = compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

print('KS metric', ks, ks < 0.09)
# print(roc_auc_truncated(y_cv, cv_predictions.predictions_1))
# split data into train and cv
X_tr, X_cv, y_tr, y_cv, X_tr_id, X_cv_id, train_predictions, cv_predictions = train_test_split(train, signal, trainids, train_predictions_all, random_state=100, test_size=0.25, shuffle=True)

# copy our predictions so they are not slices and we won't get errors
# train_predictions = train_predictions.copy()
cv_predictions = cv_predictions.copy()
X_cv = X_cv.copy()

# train on whole data set now
train_predictions = train_predictions_all.copy()
X_tr = train.copy()
y_tr = signal
X_tr = X_tr.copy()
def add_lines(data):
    data['line1'] = 1.0*np.tanh(((((((((data["IPSig"]) + (data["ISO_SumBDT"]))) - (np.minimum(((-2.0)), ((data["ISO_SumBDT"])))))) / (data["ISO_SumBDT"]))) / (np.minimum((((-1.0*((data["ISO_SumBDT"]))))), ((data["IPSig"]))))))
    data['line2'] = 1.0*np.tanh((-1.0*((((data["iso"]) + (((((((((((((data["VertexChi2"]) + ((3.0)))) / (data["ISO_SumBDT"]))) * (data["IP"]))) * 2.0)) / (data["ISO_SumBDT"]))) * (((((((((((data["VertexChi2"]) + ((3.0)))) / (data["ISO_SumBDT"]))) * (data["IP"]))) * 2.0)) / (data["ISO_SumBDT"]))))))))))
    data['line3'] = 1.0*np.tanh((-1.0*(((((((((data["IPSig"]) * ((((data["iso"]) + (((data["IP"]) * 2.0)))/2.0)))) + (np.tanh((data["p0_IsoBDT"]))))/2.0)) * ((((data["p0_IsoBDT"]) + (data["IPSig"]))/2.0)))))))
    data['line4'] = 1.0*np.tanh(((np.minimum(((np.cos((((np.cos((((data["p0_track_Chi2Dof"]) * (np.cos((data["p0_track_Chi2Dof"]))))))) * (np.log((data["IP_p0p2"])))))))), ((np.cos((data["p0_track_Chi2Dof"])))))) * (data["p0_track_Chi2Dof"])))
    data['line5'] = 1.0*np.tanh((((((((((p1)) / (((((p1)) + (((((data["SPDhits"]) / 2.0)) / 2.0)))/2.0)))) - (data["IP"]))) - (((data["SPDhits"]) / (data["p1_pt"]))))) * 2.0))
    data['line6'] = 1.0*np.tanh((((((((((((((data["CDF3"]) / (data["dira"]))) > (data["CDF3"]))*1.)) > (data["CDF3"]))*1.)) / 2.0)) + ((-1.0*((((((data["CDF3"]) * (data["p2_track_Chi2Dof"]))) * (((data["CDF3"]) * (data["p2_track_Chi2Dof"])))))))))/2.0))
    data['line7'] = 1.0*np.tanh((((-1.0*((((data["DOCAthree"]) / (data["CDF2"])))))) + (np.minimum(((((data["p2_pt"]) / (data["p0_p"])))), ((np.minimum(((data["CDF2"])), ((((np.sin((p3))) / 2.0))))))))))
    data['line8'] = 1.0*np.tanh(np.minimum((((-1.0*(((((((data["FlightDistance"]) < (data["IPSig"]))*1.)) / 2.0)))))), ((((np.minimum(((np.cos((np.log((data["p0_pt"])))))), ((np.cos((data["p1_track_Chi2Dof"])))))) / (p2))))))
    data['line9'] = 1.0*np.tanh(((np.sin((np.where(data["iso"]>0, ((((data["iso"]) - ((-1.0*((((data["IPSig"]) / 2.0))))))) / 2.0), ((((3.0) * (data["IP"]))) * 2.0) )))) / 2.0))
    data['line10'] = 1.0*np.tanh(((((np.cos(((((data["ISO_SumBDT"]) + (p2))/2.0)))) - (np.sin((np.log((data["p1_eta"]))))))) - ((((((data["ISO_SumBDT"]) + (np.cos((data["p2_IsoBDT"]))))/2.0)) * ((((data["ISO_SumBDT"]) + (np.cos((data["p2_IsoBDT"]))))/2.0))))))
    
    return data
X_tr = add_lines(X_tr)
X_cv = add_lines(X_cv)
test = add_lines(test)
train = add_lines(train)
check_agreement = add_lines(check_agreement)
feature_names = ['LifeTime',
 'dira',
 'FlightDistance',
 'FlightDistanceError',
 'IP',
 'IPSig',
 'VertexChi2',
 'pt',
 'iso',
 'ISO_SumBDT',
 'NEW_FD_SUMP',
 'NEW5_lt',
 'p_track_Chi2Dof_MAX',
 'flight_dist_sig2',
 'flight_dist_sig',
 'NEW_IP_dira',
 'p0p2_ip_ratio',
 'p1p2_ip_ratio',
 'DCA_MAX',
 'iso_bdt_min',
 'iso_min',
 'NEW_iso_abc',
 'NEW_iso_def',
 'NEW_pN_IP',
 'NEW_pN_p',
 'NEW_IP_pNpN',
 'NEW_pN_IPSig',
 'NEW_FD_LT',
                
                'line1', 'line2', 'line3', 'line4',
                 'line6',
                 'line7',
                 'line9',
                 'line10',
                 'line8',
                ]
# use the full training set for our cv, and then train only on training set so we can validate
# train_all = lgb.Dataset(train[feature_names],signal)
train_all = train_set = lgb.Dataset(X_tr[feature_names],y_tr)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 2**8,
    'metric': {'auc'},
    'min_data_in_leaf': 31,
    'max_depth': 12,
    'learning_rate': 0.05,
    'bagging_fraction': 0.5,
    'lambda': 0.1,
    'feature_fraction': 0.5,
}

cv_output = lgb.cv(
    params,
    train_all,
    num_boost_round=450,
    nfold=5,
)

best_niter = np.argmax(cv_output['auc-mean'])
best_score = cv_output['auc-mean'][best_niter]
print('Best number of iterations: {}'.format(best_niter))
print('Best CV score: {}'.format(best_score))
model = lgb.train(params, train_set, num_boost_round=best_niter)

train_predictions['predictions_2'] = model.predict(X_tr[feature_names])
cv_predictions['predictions_2'] = model.predict(X_cv[feature_names])
test_predictions['predictions_2'] = model.predict(test[feature_names])
ca_predictions['predictions_2'] = model.predict(check_agreement[feature_names])
agreement_probs = ca_predictions.predictions_2.values

ks = compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

print('KS metric', ks, ks < 0.09)
print(roc_auc_truncated(y_cv, cv_predictions['predictions_2']))
# depth 12
print('KS metric', ks, ks < 0.09)
print(roc_auc_truncated(y_cv, cv_predictions['predictions_2']))
et_features = ['LifeTime', 'dira', 'FlightDistance', 'FlightDistanceError', 'IP',
       'VertexChi2', 'pt', 'DOCAone', 'DOCAtwo', 'DOCAthree',
       'IP_p0p2', 'IP_p1p2', 'isolationa', 'isolationb', 'isolationc',
       'isolationd', 'isolatione', 'isolationf', 'iso',
       'ISO_SumBDT', 'p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT', 'p0_track_Chi2Dof',
       'p1_track_Chi2Dof', 'p2_track_Chi2Dof', 'p0_IP', 'p1_IP', 'p2_IP',
       'p0_IPSig', 'p1_IPSig', 'p2_IPSig', 'p0_pt', 'p1_pt', 'p2_pt', 'p0_p',
       'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta', 'NEW_FD_SUMP',
       'NEW5_lt', 'p_track_Chi2Dof_MAX', 'flight_dist_sig2', 'flight_dist_sig',
       'NEW_IP_dira', 'p0p2_ip_ratio', 'p1p2_ip_ratio', 'DCA_MAX',
       'iso_bdt_min', 'iso_min', 'NEW_iso_abc', 'NEW_iso_def', 'NEW_pN_IP',
       'NEW_pN_p', 'NEW_IP_pNpN', 'NEW_pN_IPSig', 'NEW_FD_LT', 
       'lines', 'line1', 'line2', 'line3', 'line4',  'line6', 'line7', # 'line5',
       'line8', 'line9', 'line10']
et = ExtraTreesClassifier(n_estimators=100, random_state=0, max_depth=22, min_impurity_decrease=1e-8, min_samples_leaf=15, n_jobs=-1, verbose=1)
et.fit(X_tr[et_features], y_tr)
tr_predictions_3 = et.predict_proba(X_tr[et_features])[:,1]
cv_predictions_3 = et.predict_proba(X_cv[et_features])[:,1]
test_predictions_3 = et.predict_proba(test[et_features])[:,1]
agreement_predictions_3 = et.predict_proba(check_agreement[et_features])[:,1]

print("Train Max:", np.max(tr_predictions_3))
print("Test Max:", np.max(test_predictions_3))
agreement_probs = agreement_predictions_3

ks = compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

noise = np.random.normal(0,0.01,len(cv_predictions_3))

print('KS metric', ks, ks < 0.09)
print(roc_auc_truncated(y_cv, cv_predictions_3))
print('KS metric', ks, ks < 0.09)
print(roc_auc_truncated(y_cv, cv_predictions_3))
train_predictions['predictions_3'] = tr_predictions_3
cv_predictions['predictions_3'] = cv_predictions_3
test_predictions['predictions_3'] = test_predictions_3
ca_predictions['predictions_3'] = agreement_predictions_3
ranked = feature_importance(et, test[et_features])
rf_features = ['lines',
 'line3',
 'IPSig',
 'p0p2_ip_ratio',
 'line2',
 'IP',
 'line8',
 'p_track_Chi2Dof_MAX',
 'dira',
 'p1p2_ip_ratio',
 'VertexChi2',
 'line9',
 'p0_track_Chi2Dof',
 'iso_bdt_min',
 'ISO_SumBDT',
 'NEW_FD_SUMP',
 'DCA_MAX',
 'p0_IP',
 'p0_IPSig',
 'flight_dist_sig2',
 'NEW_IP_dira',
 'LifeTime',
 'flight_dist_sig',
 'NEW_pN_IPSig',
 'p0_IsoBDT',
 'NEW_pN_IP',
 'line1',
 'line4',
 'NEW_pN_p',
 'p2_IPSig',
 'p1_track_Chi2Dof',
 'p1_IsoBDT',
 'NEW_FD_LT',
 'IP_p1p2',
 'pt',
 'NEW5_lt',
 'line7',
 'iso',
 'p2_track_Chi2Dof',
 'IP_p0p2',
 'p1_p',
 'p1_IPSig',
 'p2_IsoBDT',
 'p0_p',
 'p1_eta',
 'line10',
 'p2_IP',
 'NEW_IP_pNpN',
 'DOCAone',
 'p0_pt',
 'FlightDistance',
 'DOCAthree',
 'p1_pt',
 'p0_eta',
 'p1_IP',
 'FlightDistanceError',
 'DOCAtwo',
 'p2_pt',
 'p2_eta',
 'p2_p',
 'CDF1',
 'CDF2']
rf = RandomForestClassifier(n_estimators=200, random_state=0, max_depth=15, min_impurity_decrease=1e-6, min_samples_leaf=20, n_jobs=-1, verbose=1)
rf.fit(X_tr[rf_features], y_tr)
tr_predictions_4 = rf.predict_proba(X_tr[rf_features])[:,1]
cv_predictions_4 = rf.predict_proba(X_cv[rf_features])[:,1]
test_predictions_4 = rf.predict_proba(test[rf_features])[:,1]
agreement_predictions_4 = rf.predict_proba(check_agreement[rf_features])[:,1]
agreement_probs = agreement_predictions_4

ks = compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

print('KS metric', ks, ks < 0.09)
print(roc_auc_truncated(y_cv, cv_predictions_4))
print('KS metric', ks, ks < 0.09)
print(roc_auc_truncated(y_cv, cv_predictions_4))
train_predictions['predictions_4'] = tr_predictions_4
cv_predictions['predictions_4'] = cv_predictions_4
test_predictions['predictions_4'] = test_predictions_4
ca_predictions['predictions_4'] = agreement_predictions_4
ranked = feature_importance(rf, test[rf_features])
ugbc_features = ['LifeTime',
 'dira',
 'FlightDistance',
 'FlightDistanceError',
 'IP',
 'IPSig',
 'VertexChi2',
 'pt',
 'iso',
 'ISO_SumBDT',
 'NEW_FD_SUMP',
 'NEW5_lt',
 'p_track_Chi2Dof_MAX',
 'flight_dist_sig2',
 'flight_dist_sig',
 'NEW_IP_dira',
 'p0p2_ip_ratio',
 'p1p2_ip_ratio',
 'DCA_MAX',
 'iso_bdt_min',
 'iso_min',
 'NEW_iso_abc',
 'NEW_iso_def',
 'NEW_pN_IP',
 'NEW_pN_p',
 'NEW_IP_pNpN',
 'NEW_pN_IPSig',
 'NEW_FD_LT', 
 
 'lines',
 'line3',
]
loss = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0 , fl_coefficient=15, power=2)
ugbc = UGradientBoostingClassifier(loss=loss, n_estimators=250,
                                 max_depth=8,
                                 learning_rate=0.15,
                                 train_features=ugbc_features,
                                 subsample=0.7,
                                 random_state=123)

ugbc.fit(X_tr[ugbc_features + ['mass']], y_tr)
tr_predictions_5 = ugbc.predict_proba(X_tr[ugbc_features])[:,1]
cv_predictions_5 = ugbc.predict_proba(X_cv[ugbc_features])[:,1]
test_predictions_5 = ugbc.predict_proba(test[ugbc_features])[:,1]
agreement_predictions_5 = ugbc.predict_proba(check_agreement[ugbc_features])[:,1]
agreement_probs = agreement_predictions_5

ks = compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

print('KS metric', ks, ks < 0.09)
print(roc_auc_truncated(y_cv, cv_predictions_5))
# without lines
print('KS metric', ks, ks < 0.09)
print(roc_auc_truncated(y_cv, cv_predictions_5))
train_predictions['predictions_5'] = tr_predictions_5
cv_predictions['predictions_5'] = cv_predictions_5
test_predictions['predictions_5'] = test_predictions_5
ca_predictions['predictions_5'] = agreement_predictions_5
test_predictions[['id', 'predictions_5']].to_csv("20180720_ugbc_1.csv", index=False, header=["id", "prediction"])
avg_drop_cols = ["id"]
test_predictions['avg_preds'] = test_predictions.drop(avg_drop_cols, axis=1).mean(axis=1)
ca_predictions['avg_preds'] = ca_predictions.drop(avg_drop_cols, axis=1).mean(axis=1)
cv_predictions['avg_preds'] = cv_predictions.drop(avg_drop_cols, axis=1).mean(axis=1)
train_predictions['avg_preds'] = train_predictions.drop(avg_drop_cols, axis=1).mean(axis=1)
agreement_probs = ca_predictions['avg_preds']

ks = compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

print('KS metric', ks, ks < 0.09)
print(roc_auc_truncated(y_cv, cv_predictions['avg_preds']))
print('KS metric', ks, ks < 0.09)
print(roc_auc_truncated(y_cv, cv_predictions['avg_preds']))
test_predictions[['id', 'avg_preds']].to_csv("20180720_averaged_1.csv", index=False, header=["id", "prediction"])
