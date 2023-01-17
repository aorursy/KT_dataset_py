%pylab inline
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
from sklearn.model_selection import train_test_split
from hep_ml.gradientboosting import UGradientBoostingClassifier
from hep_ml.losses import BinFlatnessLossFunction
from sklearn.metrics import roc_curve, roc_auc_score, auc
from hep_ml import metrics
from sklearn.utils.validation import column_or_1d
print("Imports added...")
# -------------- loading data files -------------- #
print("Load the train/test/eval data using pandas")
train_ugbc = pd.read_csv("../input/training.csv")
train_ugbc = train_ugbc[train_ugbc['min_ANNmuon'] > 0.4]
test_ugbc  = pd.read_csv("../input/test.csv")
check_agreement = pd.read_csv('../input/check_agreement.csv', index_col='id')

trainids = train_ugbc.index.values
testids = test_ugbc.index.values
caids = check_agreement.index.values
trainsignals = train_ugbc.signal.ravel()
signal = train_ugbc.signal
print("Data loaded...")
# control switches
DO_5_LINES = True    # Do 5 lines model?
DO_5_ENS   = False     # Ensemble 5 lines model
DO_IMP = False       # Do feature permutation importances?
DO_GRAMOLIN_IMP = False   # Basis for my current 'best'. Use this as ensemble for ensemble importances.
DO_MASS_PLOT = True # Plot mass correlation?
DO_MASS_CORR = True  # Check mass correlation?
DO_NOISE = True     # Add noise to improve monte carlo vs. real. My rough understanding is this means you are training more to 
                     # predicting monte carlo rather than signal? 
DO_GRAMOLIN = True   # Include gramolin solution in final ensemble
# Constants
MC = 0.002        # Competition allowed mass correlation
def get_ks_metric(df_agree, df_test):
    sig_ind = df_agree[df_agree['signal'] == 1].index
    bck_ind = df_agree[df_agree['signal'] == 0].index

    mc_prob = numpy.array(df_test.loc[sig_ind]['prediction'])
    mc_weight = numpy.array(df_agree.loc[sig_ind]['weight'])
    data_prob = numpy.array(df_test.loc[bck_ind]['prediction'])
    data_weight = numpy.array(df_agree.loc[bck_ind]['weight'])
    val, agreement_metric = check_agreement_ks_sample_weighted(data_prob, mc_prob, data_weight, mc_weight)
    return agreement_metric['ks']
def check_agreement_ks_sample_weighted (data_prediction, mc_prediction, weights_data, weights_mc):
    data_prediction, weights_data = map(column_or_1d, [data_prediction, weights_data])
    mc_prediction, weights_mc = map(column_or_1d, [mc_prediction, weights_mc])

    assert numpy.all(data_prediction >= 0.) and numpy.all(data_prediction <= 1.), 'error in prediction'
    assert numpy.all(mc_prediction >= 0.) and numpy.all(mc_prediction <= 1.), 'error in prediction'

    weights_data = weights_data / numpy.sum(weights_data)
    weights_mc = weights_mc / numpy.sum(weights_mc)

    data_neg = data_prediction[weights_data < 0]
    weights_neg = -weights_data[weights_data < 0]
    mc_prediction = numpy.concatenate((mc_prediction, data_neg))
    weights_mc = numpy.concatenate((weights_mc, weights_neg))
    data_prediction = data_prediction[weights_data >= 0]
    weights_data = weights_data[weights_data >= 0]

    assert numpy.all(weights_data >= 0) and numpy.all(weights_mc >= 0)
    assert numpy.allclose(weights_data.sum(), weights_mc.sum())

    weights_data /= numpy.sum(weights_data)
    weights_mc /= numpy.sum(weights_mc)

    fpr, tpr, _ = roc_curve_splitted(data_prediction, mc_prediction, weights_data, weights_mc)

    Dnm = numpy.max(numpy.abs(fpr - tpr))
    Dnm_part = numpy.max(numpy.abs(fpr - tpr)[fpr + tpr < 1])

    result = {'ks': Dnm, 'ks_part': Dnm_part}
    return Dnm_part < 0.03, result
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
def check_correlation(probabilities, mass):
    probabilities, mass = map(column_or_1d, [probabilities, mass])

    y_pred = numpy.zeros(shape=(len(probabilities), 2))
    y_pred[:, 1] = probabilities
    y_pred[:, 0] = 1 - probabilities
    y_true = [0] * len(probabilities)
    df_mass = pd.DataFrame({'mass': mass})
    cvm = metrics.BinBasedCvM(uniform_features=['mass'], uniform_label=0)
    cvm.fit(df_mass, y_true)
    return cvm(y_true, y_pred, sample_weight=None)

df_agreement = pd.read_csv('../input/check_agreement.csv')
df_corr_check = pd.read_csv("../input/check_correlation.csv")
# Physical constants:
c = 299.792458     # Speed of light
m_mu = 105.6583715 # Muon mass (in MeV)
m_tau = 1776.82    # Tau mass (in MeV)

# List of the features for the first booster:
list1 = [
# Original features:
         'FlightDistance',
         'FlightDistanceError',
         'LifeTime',
         'IP',
         'IPSig',
         'VertexChi2',
         'dira',
         'pt',
         'DOCAone',
         'DOCAtwo',
         'DOCAthree',
         'IP_p0p2',
         'IP_p1p2',
         'isolationa',
         'isolationb',
         'isolationc',
         'isolationd',
         'isolatione',
         'isolationf',
         'iso',
         'CDF1',
         'CDF2',
         'CDF3',
         'ISO_SumBDT',
         'p0_IsoBDT',
         'p1_IsoBDT',
         'p2_IsoBDT',
         'p0_track_Chi2Dof',
         'p1_track_Chi2Dof',
         'p2_track_Chi2Dof',
         'p0_IP',
         'p0_IPSig',
         'p1_IP',
         'p1_IPSig',
         'p2_IP',
         'p2_IPSig',
# Extra features:
         'E',
         'FlightDistanceSig',
         'DOCA_sum',
         'isolation_sum',
         'IsoBDT_sum',
         'track_Chi2Dof',
         'IP_sum',
         'IPSig_sum',
         'CDF_sum'
        ]

# List of the features for the second booster:
list2 = [
# Original features:
         'dira',
         'pt',
         'p0_pt',
         'p0_p',
         'p0_eta',
         'p1_pt',
         'p1_p',
         'p1_eta',
         'p2_pt',
         'p2_p',
         'p2_eta',
# Extra features:
         'E',
         'pz',
         'beta',
         'gamma',
         'beta_gamma',
         'Delta_E',
         'Delta_M',
         'flag_M',
         'E0',
         'E1',
         'E2',
         'E0_ratio',
         'E1_ratio',
         'E2_ratio',
         'p0_pt_ratio',
         'p1_pt_ratio',
         'p2_pt_ratio',
         'eta_01',
         'eta_02',
         'eta_12',
         't_coll'
         ]

# Function to add extra features:
def add_features_gramolin(df):
  
  # Number of events:
  N = len(df)
  
  # Internal arrays:
  p012_p = np.zeros(3)
  p012_pt = np.zeros(3)
  p012_z = np.zeros(3)
  p012_eta = np.zeros(3)
  p012_IsoBDT = np.zeros(3)
  p012_track_Chi2Dof = np.zeros(3)
  p012_IP = np.zeros(3)
  p012_IPSig = np.zeros(3)
  CDF123 = np.zeros(3)
  isolation = np.zeros(6)
  
  # Kinematic features related to the mother particle:
  E = np.zeros(N)
  pz = np.zeros(N)
  beta = np.zeros(N)
  gamma = np.zeros(N)
  beta_gamma = np.zeros(N)
  M_lt = np.zeros(N)
  M_inv = np.zeros(N)
  Delta_E = np.zeros(N)
  Delta_M = np.zeros(N)
  flag_M = np.zeros(N)
  
  # Kinematic features related to the final-state particles p0, p1, and p2:
  E012 = np.zeros((N,3))
  E012_ratio = np.zeros((N,3))
  p012_pt_ratio = np.zeros((N,3))
  eta_01 = np.zeros(N)
  eta_02 = np.zeros(N)
  eta_12 = np.zeros(N)
  t_coll = np.zeros(N)
  
  # Other extra features:
  FlightDistanceSig = np.zeros(N)
  DOCA_sum = np.zeros(N)
  isolation_sum = np.zeros(N)
  IsoBDT_sum = np.zeros(N)
  track_Chi2Dof = np.zeros(N)
  IP_sum = np.zeros(N)
  IPSig_sum = np.zeros(N)
  CDF_sum = np.zeros(N)
  
  for i in range(N):
    # Read some of the original features:  
    pt = df['pt'].values[i]
    dira = df['dira'].values[i]
    LifeTime = df['LifeTime'].values[i]
    FlightDistance = df['FlightDistance'].values[i]
    FlightDistanceError = df['FlightDistanceError'].values[i]
    DOCAone = df['DOCAone'].values[i]
    DOCAtwo = df['DOCAtwo'].values[i]
    DOCAthree = df['DOCAthree'].values[i]
    isolation[0] = df['isolationa'].values[i]
    isolation[1] = df['isolationb'].values[i]
    isolation[2] = df['isolationc'].values[i]
    isolation[3] = df['isolationd'].values[i]
    isolation[4] = df['isolatione'].values[i]
    isolation[5] = df['isolationf'].values[i]
    
    for j in range(3):
      p012_p[j] = df['p'+str(j)+'_p'].values[i]
      p012_pt[j] = df['p'+str(j)+'_pt'].values[i]
      p012_eta[j] = df['p'+str(j)+'_eta'].values[i]
      p012_IsoBDT[j] = df['p'+str(j)+'_IsoBDT'].values[i]
      p012_track_Chi2Dof[j] = df['p'+str(j)+'_track_Chi2Dof'].values[i]
      p012_IP[j] = df['p'+str(j)+'_IP'].values[i]
      p012_IPSig[j] = df['p'+str(j)+'_IPSig'].values[i]
      CDF123[j] = df['CDF'+str(j+1)].values[i]
    
    # Differences between pseudorapidities of the final-state particles:
    eta_01[i] = p012_eta[0] - p012_eta[1]
    eta_02[i] = p012_eta[0] - p012_eta[2]
    eta_12[i] = p012_eta[1] - p012_eta[2]
    
    # Transverse collinearity of the final-state particles (equals to 1 if they are collinear):
    t_coll[i] = sum(p012_pt[:])/pt
    
    # Longitudinal momenta of the final-state particles:
    p012_z[:] = p012_pt[:]*np.sinh(p012_eta[:])
    
    # Energies of the final-state particles:
    E012[i,:] = np.sqrt(np.square(m_mu) + np.square(p012_p[:]))
    
    # Energy and momenta of the mother particle:
    E[i] = sum(E012[i,:])
    pz[i] = sum(p012_z[:])
    p = np.sqrt(np.square(pt) + np.square(pz[i]))
    
    # Energies and momenta of the final-state particles relative to those of the mother particle:
    E012_ratio[i,:] = E012[i,:]/E[i]
    p012_pt_ratio[i,:] = p012_pt[:]/pt
    
    # Mass of the mother particle calculated from FlightDistance and LifeTime:
    beta_gamma[i] = FlightDistance/(LifeTime*c)
    M_lt[i] = p/beta_gamma[i]
    
    # If M_lt is around the tau mass then flag_M = 1 (otherwise 0):
    if np.fabs(M_lt[i] - m_tau - 1.44) < 17: flag_M[i] = 1
    
    # Invariant mass of the mother particle calculated from its energy and momentum:        
    M_inv[i] = np.sqrt(np.square(E[i]) - np.square(p))
    if (np.isnan(M_inv[i])):      # mjh for about 11 records this is true
        M_inv[i] = 0
        gamma[i] = 0
        beta[i] = 0
    else:
        # Relativistic gamma and beta of the mother particle:
        gamma[i] = E[i]/M_inv[i]   
        beta[i] = np.sqrt(np.square(gamma[i]) - 1.)/gamma[i]
    
    # Difference between M_lt and M_inv:
    Delta_M[i] = M_lt[i] - M_inv[i]
    
    # Difference between energies of the mother particle calculated in two different ways:
    Delta_E[i] = np.sqrt(np.square(M_lt[i]) + np.square(p)) - E[i]
    
    # Other extra features:
    FlightDistanceSig[i] = FlightDistance/FlightDistanceError
    DOCA_sum[i] = DOCAone + DOCAtwo + DOCAthree
    isolation_sum[i] = sum(isolation[:])
    IsoBDT_sum[i] = sum(p012_IsoBDT[:])
    track_Chi2Dof[i] = np.sqrt(sum(np.square(p012_track_Chi2Dof[:] - 1.)))
    IP_sum[i] = sum(p012_IP[:])
    IPSig_sum[i] = sum(p012_IPSig[:])
    CDF_sum[i] = sum(CDF123[:])
  
  # Kinematic features related to the mother particle:
  df['E'] = E
  df['pz'] = pz
  df['beta'] = beta
  df['gamma'] = gamma
  df['beta_gamma'] = beta_gamma
  df['M_lt'] = M_lt
  df['M_inv'] = M_inv
  df['Delta_E'] = Delta_E
  df['Delta_M'] = Delta_M
  df['flag_M'] = flag_M
  
  # Kinematic features related to the final-state particles:
  df['E0'] = E012[:,0]
  df['E1'] = E012[:,1]
  df['E2'] = E012[:,2]
  df['E0_ratio'] = E012_ratio[:,0]
  df['E1_ratio'] = E012_ratio[:,1]
  df['E2_ratio'] = E012_ratio[:,2]
  df['p0_pt_ratio'] = p012_pt_ratio[:,0]
  df['p1_pt_ratio'] = p012_pt_ratio[:,1]
  df['p2_pt_ratio'] = p012_pt_ratio[:,2]
  df['eta_01'] = eta_01
  df['eta_02'] = eta_02
  df['eta_12'] = eta_12
  df['t_coll'] = t_coll
  
  # Other extra features:
  df['FlightDistanceSig'] = FlightDistanceSig
  df['DOCA_sum'] = DOCA_sum
  df['isolation_sum'] = isolation_sum
  df['IsoBDT_sum'] = IsoBDT_sum
  df['track_Chi2Dof'] = track_Chi2Dof
  df['IP_sum'] = IP_sum
  df['IPSig_sum'] = IPSig_sum
  df['CDF_sum'] = CDF_sum
  
  return df
#--------------- feature engineering -------------- #
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
if DO_5_LINES or DO_5_ENS:
    tr_preds_1 = GP(train_ugbc).values
    test_preds_1 = GP(test_ugbc).values
    ca_preds_1 = GP(check_agreement).values

    test_predictions = pd.DataFrame({'preds_line5':test_preds_1})
    train_predictions_all = pd.DataFrame({'id':trainids,'predictions_1':tr_preds_1})
    ca_predictions = pd.DataFrame({'id':caids,'predictions_1':ca_preds_1})
# since the target is not used for this model we can add the feature to our data without any leakage
if DO_5_LINES:
    train_ugbc['lines'] = tr_preds_1
    check_agreement['lines'] = ca_preds_1
    test_ugbc['lines'] = test_preds_1
if DO_5_LINES or DO_5_ENS:
    agreement_probs = ca_predictions.predictions_1

    ks = compute_ks(
        agreement_probs[check_agreement['signal'].values == 0],
        agreement_probs[check_agreement['signal'].values == 1],
        check_agreement[check_agreement['signal'] == 0]['weight'].values,
        check_agreement[check_agreement['signal'] == 1]['weight'].values)

    print('5 line KS metric', ks, ks < 0.09)
# print(roc_auc_truncated(y_cv, cv_predictions.predictions_1))
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
if DO_5_LINES:
    train, test, y_train, y_test, train_id, test_id, train_predictions, cv_predictions = train_test_split(train_ugbc, signal, trainids, train_predictions_all, random_state=100, test_size=0.25, shuffle=True)
    cv_predictions = cv_predictions.copy()
    train_predictions = train_predictions_all.copy()
else:
    train, test, y_train, y_test, train_id, test_id = train_test_split(train_ugbc, signal, trainids, random_state=100, test_size=0.25, shuffle=True)
    
# copy our predictions so they are not slices and we won't get errors
# train_predictions = train_predictions.copy()

test = test.copy()

# train on whole data set now
train = train.copy()
y_train = signal

#train = X_tr.copy()
if DO_5_LINES:
    train = add_lines(train)
    test = add_lines(test)
    test_ugbc = add_lines(test_ugbc)
    train_ugbc = add_lines(train_ugbc)
    check_agreement = add_lines(check_agreement)
print("Add features")
train_gramolin = add_features_gramolin(train.copy())
train = add_features(train)
train_ugbc = add_features(train_ugbc)
test_gramolin = add_features_gramolin(test)
test = add_features(test)
test_ugbc = add_features(test_ugbc)
check_agreement = add_features(check_agreement)
print("features added...")
print("Eliminate features")
filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal',
              'SPDhits','CDF1', 'CDF2', 'CDF3',
              'isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt',
              'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta',
              'isolationa', 'isolationb', 'isolationc', 'isolationd', 'isolatione', 'isolationf',
              'p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT',
              'p0_IP', 'p1_IP', 'p2_IP',
              'IP_p0p2', 'IP_p1p2',
              'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof',
              'p0_IPSig', 'p1_IPSig', 'p2_IPSig',
              'DOCAone', 'DOCAtwo', 'DOCAthree',
              'lines', 'line5', 'line6']
              #'line10']
features = list(f for f in train.columns if f not in filter_out)
if DO_5_LINES:
    train_gramolin = add_lines(train_gramolin)
    test_gramolin = add_lines(test_gramolin)
    # list1.extend(['line1','line2','line3','line4','line6','line7','line8']) # ,'line9','line10'])
    list1.extend(['line1','line2','line3','line4','line6'])

# mjh - looking to update my later importance functions to include the agreement and correlation metrics
#       by feature. Maybe make this process of merging features, and passing metric constraint tests 
#       a little less trial and err. This version should be close to back to my best, except maybe that was just
#       5 lines model lines1-4 only? It appeared that way when I did a quick diff. Version 59 was I think the best.
# mjh try some of the original ugbc features and eliminate some that eliminates
#train_gramolin = add_features(train_gramolin)
#test_gramolin = add_features(test_gramolin)
#list1 = list(f for f in list1 if f not in filter_out)
#list1.extend(['NEW5_lt'])
#list2 = list(f for f in list2 if f not in filter_out)
#list2.extend(['NEW_FD_SUMP'])

loss = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0, fl_coefficient=15, power=2)
gramolin1 = UGradientBoostingClassifier(loss=loss, n_estimators=550,
                                 max_depth=6,
                                 learning_rate=0.15,
                                 train_features=list1,
                                 subsample=0.7,
                                 random_state=123)
gramolin1.fit(train_gramolin[list1 + ['mass']], train_gramolin['signal'])
y_pred_gramolin1 = gramolin1.predict_proba(test_gramolin[list1])[:, 1]
roc_auc_gramolin1 = roc_auc_score(test_gramolin['signal'], y_pred_gramolin1)
print("Gramolin 1 AUC:",roc_auc_gramolin1)
df_agreement_gramolin = add_features_gramolin(df_agreement)
df_agreement_gramolin = add_features(df_agreement_gramolin)
if DO_5_LINES:
    df_agreement_gramolin = add_lines(df_agreement_gramolin)
agreement_probs_gramolin1 = gramolin1.predict_proba(df_agreement_gramolin[list1])[:, 1]
ks = compute_ks(
    agreement_probs_gramolin1[df_agreement_gramolin['signal'].values == 0],
    agreement_probs_gramolin1[df_agreement_gramolin['signal'].values == 1],
    df_agreement_gramolin[df_agreement_gramolin['signal'] == 0]['weight'].values,
    df_agreement_gramolin[df_agreement_gramolin['signal'] == 1]['weight'].values)
df_corr_check_gramolin = add_features_gramolin(df_corr_check)
df_corr_check_gramolin = add_features(df_corr_check_gramolin)
if DO_5_LINES:
    df_corr_check_gramolin = add_lines(df_corr_check_gramolin)
y_mc = gramolin1.predict_proba(df_corr_check_gramolin[list1])[:, 1]
mc1 = check_correlation(y_mc, df_corr_check['mass'])
print ('Gramolin 1 KS metric:', ks, "is OK:", ks < 0.09,'MC metric:', mc1, "is OK:", mc1 < MC)
test_ugbc_gramolin = add_features_gramolin(test_ugbc)
test_ugbc_gramolin = add_features(test_ugbc_gramolin)
if DO_5_LINES:
    test_ugbc_gramolin = add_lines(test_ugbc_gramolin)
test_p_gramolin1 = gramolin1.predict_proba(test_ugbc_gramolin[list1])[:,1]
result1 = pd.DataFrame({'id': test_ugbc['id']})
result1['prediction'] = test_p_gramolin1 
result1.to_csv('gramolin1.csv', index=False, header=["id", "prediction"], sep=',', mode='a')
gramolin2 = UGradientBoostingClassifier(loss=loss, n_estimators=550,
                                 max_depth=6,
                                 learning_rate=0.15,
                                 train_features=list2,
                                 subsample=0.7,
                                 random_state=123)
gramolin2.fit(train_gramolin[list2 + ['mass']], train_gramolin['signal'])
y_pred_gramolin2 = gramolin2.predict_proba(test_gramolin[list2])[:, 1]
roc_auc_gramolin2 = roc_auc_score(test_gramolin['signal'], y_pred_gramolin2)
print("Gramolin 2 AUC:",roc_auc_gramolin2)
agreement_probs_gramolin2 = gramolin2.predict_proba(df_agreement_gramolin[list2])[:, 1]

ks = compute_ks(
    agreement_probs_gramolin1[df_agreement_gramolin['signal'].values == 0],
    agreement_probs_gramolin1[df_agreement_gramolin['signal'].values == 1],
    df_agreement_gramolin[df_agreement_gramolin['signal'] == 0]['weight'].values,
    df_agreement_gramolin[df_agreement_gramolin['signal'] == 1]['weight'].values)
y_mc = gramolin2.predict_proba(df_corr_check_gramolin[list1])[:, 1]
mc2 = check_correlation(y_mc, df_corr_check['mass'])
print ('Gramolin 2 KS metric:', ks, "is OK:", ks < 0.09,'MC metric:', mc2, "is OK:", mc2 < MC)
test_p_gramolin2 = gramolin2.predict_proba(test_ugbc_gramolin[list2])[:,1]
result2 = pd.DataFrame({'id': test_ugbc['id']})
result2['prediction'] = test_p_gramolin2 
result2.to_csv('gramolin2.csv', index=False, header=["id", "prediction"], sep=',', mode='a')

p_weight = 0.94   
# Weighted average of the predictions:
result = pd.DataFrame({'id': test_ugbc['id']})
result['prediction'] = 0.5*(p_weight*test_p_gramolin1 + (1 - p_weight)*test_p_gramolin2)
# Write to the submission file:
result.to_csv('gramolin_sub.csv', index=False, header=["id", "prediction"], sep=',', mode='a')
if not 'test_predictions' in locals():
    test_predictions = pd.DataFrame({'gramolin':result['prediction']})
else:
    test_predictions['gramolin'] = result['prediction']
print("Gramolin Done...")
#-------------------  UGBC model -------------------- #
print("Train a UGradientBoostingClassifier")
loss = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0 , fl_coefficient=15, power=2)
ugbc = UGradientBoostingClassifier(loss=loss, n_estimators=550,
                                 max_depth=6,
                                 learning_rate=0.15,
                                 train_features=features,
                                 subsample=0.7,
                                 random_state=123)
ugbc.fit(train[features + ['mass']], train['signal'])
print("Done...")
def plot_metrics(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    plt.plot(fpr, tpr, label='ROC AUC=%f' % roc_auc)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title("ROC Curve")
y_pred = ugbc.predict_proba(test[features])[:, 1]

plot_metrics(test['signal'], y_pred)
train.shape, y_pred.shape
if DO_MASS_CORR:
    df_corr_check = pd.read_csv("../input/check_correlation.csv")
    df_corr_check = add_features(df_corr_check)
    df_corr_check['lines'] = GP(df_corr_check).values
    df_corr_check = add_lines(df_corr_check)
if DO_MASS_PLOT:
    df_corr_check.shape
if DO_MASS_PLOT or DO_MASS_CORR:
    y_pred = ugbc.predict(df_corr_check[features])
def efficiencies(features, thresholds=None, mask=None, bins=30, labels_dict=None, ignored_sideband=0.0,
                     errors=False, grid_columns=2):
        """
        Efficiencies for spectators
        :param features: using features (if None then use classifier's spectators)
        :type features: None or list[str]
        :param bins: bins for histogram
        :type bins: int or array-like
        :param mask: mask for data, which will be used
        :type mask: None or numbers.Number or array-like or str or function(pandas.DataFrame)
        :param list[float] thresholds: thresholds on prediction
        :param bool errors: if True then use errorbar, else interpolate function
        :param labels_dict: label -- name for class label
            if None then {0: 'bck', '1': 'signal'}
        :type labels_dict: None or OrderedDict(int: str)
        :param int grid_columns: count of columns in grid
        :param float ignored_sideband: (0, 1) percent of plotting data
        :rtype: plotting.GridPlot
        """
        mask, data, class_labels, weight = self._apply_mask(
            mask, self._get_features(features), self.target, self.weight)
        labels_dict = self._check_labels(labels_dict, class_labels)

        plots = []
        for feature in data.columns:
            for name, prediction in self.prediction.items():
                prediction = prediction[mask]
                eff = OrderedDict()
                for label, label_name in labels_dict.items():
                    label_mask = class_labels == label
                    eff[label_name] = utils.get_efficiencies(prediction[label_mask, label],
                                                             data[feature][label_mask].values,
                                                             bins_number=bins,
                                                             sample_weight=weight[label_mask],
                                                             thresholds=thresholds, errors=errors,
                                                             ignored_sideband=ignored_sideband)

                for label_name, eff_data in eff.items():
                    if errors:
                        plot_fig = plotting.ErrorPlot(eff_data)
                    else:
                        plot_fig = plotting.FunctionsPlot(eff_data)
                    plot_fig.xlabel = feature
                    plot_fig.ylabel = 'Efficiency for {}'.format(name)
                    plot_fig.title = '{} flatness'.format(label_name)
                    plot_fig.ylim = (0, 1)
                    plots.append(plot_fig)

        return plotting.GridPlot(grid_columns, *plots)
def check_arrays(*arrays):
    """
    Left for consistency, version of `sklearn.validation.check_arrays`
    :param list[iterable] arrays: arrays with same length of first dimension.
    """
    assert len(arrays) > 0, 'The number of array must be greater than zero'
    checked_arrays = []
    shapes = []
    for arr in arrays:
        if arr is not None:
            checked_arrays.append(numpy.array(arr))
            shapes.append(checked_arrays[-1].shape[0])
        else:
            checked_arrays.append(None)
    assert numpy.sum(numpy.array(shapes) == shapes[0]) == len(shapes), 'Different shapes of the arrays {}'.format(
        shapes)
    return checked_arrays
def get_efficiencies(prediction, spectator, sample_weight=None, bins_number=20,
                     thresholds=None, errors=False, ignored_sideband=0.0):
    """
    Construct efficiency function dependent on spectator for each threshold
    Different score functions available: Efficiency, Precision, Recall, F1Score,
    and other things from sklearn.metrics
    :param prediction: list of probabilities
    :param spectator: list of spectator's values
    :param bins_number: int, count of bins for plot
    :param thresholds: list of prediction's threshold
        (default=prediction's cuts for which efficiency will be [0.2, 0.4, 0.5, 0.6, 0.8])
    :return:
        if errors=False
        OrderedDict threshold -> (x_values, y_values)
        if errors=True
        OrderedDict threshold -> (x_values, y_values, y_err, x_err)
        All the parts: x_values, y_values, y_err, x_err are numpy.arrays of the same length.
    """
    prediction, spectator, sample_weight = \
        check_arrays(prediction, spectator, sample_weight)

    spectator_min, spectator_max = weighted_quantile(spectator, [ignored_sideband, (1. - ignored_sideband)])
    mask = (spectator >= spectator_min) & (spectator <= spectator_max)
    spectator = spectator[mask]
    prediction = prediction[mask]
    bins_number = min(bins_number, len(prediction))
    sample_weight = sample_weight if sample_weight is None else numpy.array(sample_weight)[mask]

    if thresholds is None:
        thresholds = [weighted_quantile(prediction, quantiles=1 - eff, sample_weight=sample_weight)
                      for eff in [0.2, 0.4, 0.5, 0.6, 0.8]]

    binner = Binner(spectator, bins_number=bins_number)
    if sample_weight is None:
        sample_weight = numpy.ones(len(prediction))
    bins_data = binner.split_into_bins(spectator, prediction, sample_weight)

    bin_edges = numpy.array([spectator_min] + list(binner.limits) + [spectator_max])
    xerr = numpy.diff(bin_edges) / 2.
    result = OrderedDict()
    for threshold in thresholds:
        x_values = []
        y_values = []
        N_in_bin = []
        for num, (masses, probabilities, weights) in enumerate(bins_data):
            y_values.append(numpy.average(probabilities > threshold, weights=weights))
            N_in_bin.append(numpy.sum(weights))
            if errors:
                x_values.append((bin_edges[num + 1] + bin_edges[num]) / 2.)
            else:
                x_values.append(numpy.mean(masses))

        x_values, y_values, N_in_bin = check_arrays(x_values, y_values, N_in_bin)
        if errors:
            result[threshold] = (x_values, y_values, numpy.sqrt(y_values * (1 - y_values) / N_in_bin), xerr)
        else:
            result[threshold] = (x_values, y_values)
    return result
def weighted_quantile(array, quantiles, sample_weight=None, array_sorted=False, old_style=False):
    """Computing quantiles of array. Unlike the numpy.percentile, this function supports weights,
    but it is inefficient and performs complete sorting.
    :param array: distribution, array of shape [n_samples]
    :param quantiles: floats from range [0, 1] with quantiles of shape [n_quantiles]
    :param sample_weight: optional weights of samples, array of shape [n_samples]
    :param array_sorted: if True, the sorting step will be skipped
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: array of shape [n_quantiles]
    Example:
    >>> weighted_quantile([1, 2, 3, 4, 5], [0.5])
    Out: array([ 3.])
    >>> weighted_quantile([1, 2, 3, 4, 5], [0.5], sample_weight=[3, 1, 1, 1, 1])
    Out: array([ 2.])
    """
    array = numpy.array(array)
    quantiles = numpy.array(quantiles)
    sample_weight = check_sample_weight(array, sample_weight)
    assert numpy.all(quantiles >= 0) and numpy.all(quantiles <= 1), 'Percentiles should be in [0, 1]'

    if not array_sorted:
        array, sample_weight = reorder_by_first(array, sample_weight)

    weighted_quantiles = numpy.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= numpy.sum(sample_weight)
    return numpy.interp(quantiles, weighted_quantiles, array)


def check_sample_weight(y_true, sample_weight):
    """Checks the weights, if None, returns array.
    :param y_true: labels (or any array of length [n_samples])
    :param sample_weight: None or array of length [n_samples]
    :return: numpy.array of shape [n_samples]
    """
    if sample_weight is None:
        return numpy.ones(len(y_true), dtype=numpy.float)
    else:
        sample_weight = numpy.array(sample_weight, dtype=numpy.float)
        assert len(y_true) == len(sample_weight), \
            "The length of weights is different: not {0}, but {1}".format(len(y_true), len(sample_weight))
        return sample_weight



def reorder_by_first(*arrays):
    """
    Applies the same permutation to all passed arrays,
    permutation sorts the first passed array
    """
    arrays = check_arrays(*arrays)
    order = numpy.argsort(arrays[0])
    return [arr[order] for arr in arrays]

class Binner(object):
    def __init__(self, values, bins_number):
        """
        Binner is a class that helps to split the values into several bins.
        Initially an array of values is given, which is then splitted into 'bins_number' equal parts,
        and thus we are computing limits (boundaries of bins).
        """
        percentiles = [i * 100.0 / bins_number for i in range(1, bins_number)]
        self.limits = numpy.percentile(values, percentiles)

    def get_bins(self, values):
        """Given the values of feature, compute the index of bin
        :param values: array of shape [n_samples]
        :return: array of shape [n_samples]
        """
        return numpy.searchsorted(self.limits, values)

    def set_limits(self, limits):
        """Change the thresholds inside bins."""
        self.limits = limits

    @property
    def bins_number(self):
        """:return: number of bins"""
        return len(self.limits) + 1

    def split_into_bins(self, *arrays):
        """
        :param arrays: data to be splitted, the first array corresponds
        :return: sequence of length [n_bins] with values corresponding to each bin.
        """
        values = arrays[0]
        for array in arrays:
            assert len(array) == len(values), "passed arrays have different length"
        bins = self.get_bins(values)
        result = []
        for bin in range(len(self.limits) + 1):
            indices = bins == bin
            result.append([numpy.array(array)[indices] for array in arrays])
        return result
from collections import OrderedDict


if DO_MASS_PLOT:
    eff = get_efficiencies(y_pred, df_corr_check.mass, thresholds=[0.5]) #, thresholds=[0.2, 0.4, 0.5, 0.6, 0.8])
if DO_MASS_PLOT:
    eff.keys()
if DO_MASS_PLOT:
    for label_name, eff_data in eff.items():
        pyplot.plot(eff_data[0], eff_data[1], label="global eff  %.1f" % label_name)
    pyplot.xlabel('mass')
    pyplot.ylabel('Efficiency')
    pyplot.legend();
if DO_MASS_CORR:
    corr_metric = check_correlation(y_pred, df_corr_check['mass'])
    print (corr_metric)
df_agreement = add_features(df_agreement)
df_agreement['lines'] = GP(df_agreement).values
df_agreement = add_lines(df_agreement)
df_agreement.shape
df_agreement.columns
df_agreement[features].head()
agreement_probs = ugbc.predict_proba(df_agreement[features])[:, 1]

ks = compute_ks(
    agreement_probs[df_agreement['signal'].values == 0],
    agreement_probs[df_agreement['signal'].values == 1],
    df_agreement[df_agreement['signal'] == 0]['weight'].values,
    df_agreement[df_agreement['signal'] == 1]['weight'].values)
print ('UGBC KS metric:', ks, "is OK:", ks < 0.09)
def plot_ks(X_agreement, y_pred):
    sig_ind = X_agreement[X_agreement['signal'] == 1].index
    bck_ind = X_agreement[X_agreement['signal'] == 0].index

    mc_prob = y_pred[sig_ind]
    mc_weight = numpy.array(X_agreement.loc[sig_ind]['weight'])
    data_prob = y_pred[bck_ind]
    data_weight = numpy.array(X_agreement.loc[bck_ind]['weight'])
    inds = data_weight < 0
    mc_weight = numpy.array(list(mc_weight) + list(-data_weight[inds]))
    mc_prob = numpy.array(list(mc_prob) + list(data_prob[inds]))
    data_prob = data_prob[data_weight >= 0]
    data_weight = data_weight[data_weight >= 0]
    hist(data_prob, weights=data_weight, color='r', histtype='step', normed=True, bins=60, label='data')
    hist(mc_prob, weights=mc_weight, color='b', histtype='step', normed=True, bins=60, label='mc')
    xlabel("prediction")
    legend(loc=2)
    show()
plot_ks(df_agreement, agreement_probs)
def add_noise(array, level=0.40, random_seed=34):
    numpy.random.seed(random_seed)
    return level * numpy.random.random(size=array.size) + (1 - level) * array
if DO_NOISE:
    agreement_probs_noise = add_noise(ugbc.predict_proba(df_agreement[features])[:, 1])
if DO_NOISE:
    ks_noise = compute_ks(
        agreement_probs_noise[df_agreement['signal'].values == 0],
        agreement_probs_noise[df_agreement['signal'].values == 1],
        df_agreement[df_agreement['signal'] == 0]['weight'].values,
        df_agreement[df_agreement['signal'] == 1]['weight'].values)
    print ('KS metric:', ks_noise, "is OK:", ks_noise < 0.09)
if DO_NOISE:
    plot_ks(df_agreement, agreement_probs_noise)
if DO_NOISE:
    test.shape
if DO_NOISE:
    y_pred = add_noise(ugbc.predict_proba(test[features])[:, 1])

    plot_metrics(test['signal'], y_pred)
    test.shape, y_pred.shape
def ensemble_preds(preds):
    """
    Take the mean or apply whatever weighting you want to the passed predictions.
    Returns the single ensembled prediction set
    """
    p_weight = 0.94
    return 0.5*(p_weight*preds[0,] + (1 - p_weight)*preds[1,])
def ensemble_metric(models, X_train, y_train, features):
    preds = []
    for model in models:
        preds.append(model.predict_proba(X_train[features])[:, 1])
    ens_preds = ensemble_preds(preds)
    roc_auc = roc_auc_truncated(y_train, ens_preds)
    return roc_auc
def metric(model, X_train, y_train, features):
    y_pred = model.predict_proba(X_train[features])[:, 1]
    y_agree = model.predict_proba(df_agreement[features])[:, 1]
    y_corr = model.predict_proba(df_corr_check[features])[:, 1]
#    roc_auc = roc_auc_score(y_train, y_pred) 
# use truncated since thats what evaluation is actually done on
    roc_auc = roc_auc_truncated(y_train, y_pred)
    ks_imp = compute_ks(
        y_agree[df_agreement['signal'].values == 0],
        y_agree[df_agreement['signal'].values == 1],
        df_agreement[df_agreement['signal'] == 0]['weight'].values,
        df_agreement[df_agreement['signal'] == 1]['weight'].values)
    corr_metric = check_correlation(y_corr, df_corr_check['mass'])
    return roc_auc, ks_imp, corr_metric
def perm_ens_imp(models, X_train, y_train, feature_sets):
    base_auc, base_agree, base_corr = ensemble_metric(model, X_train, y_train, feature_sets)
    print("Baseline = ",base_auc,"KS metric:",base_agree,"MC metric:",base_corr)
    imp_auc = [], imp_agree = [], imp_corr = []
    for features in feature_sets:
        for col in features:
            save = X_train[col].copy()
            X_train[col] = np.random.permutation(X_train[col])
            m_auc, m_agree, m_corr = metric(model, X_train, y_train, features)
            X_train[col] = save
            imp_auc.append(base_auc - m_auc)
            imp_agree.append(base_agree - m_agree)
            imp_corr.append(base_corr - m_corr)
    return np.array([imp_auc, imp_agree, imp_corr])    
def permutation_importances(model, X_train, y_train, features):
    base_auc, base_agree, base_corr = metric(model, X_train, y_train, features)
    print("Baseline = ",base_auc,"KS metric:",base_agree,"MC metric:",base_corr)
    imp_auc = [], imp_agree = [], imp_corr = []
    for col in features:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m_auc, m_agree, m_corr = metric(model, X_train, y_train, features)
        X_train[col] = save
        imp_auc.append(base_auc - m_auc)
        imp_agree.append(base_agree - m_agree)
        imp_corr.append(base_corr - m_corr)
    return np.array([imp_auc, imp_agree, imp_corr])
if DO_IMP:
    print("Importance preparation")
    filter_imp = ['id', 'min_ANNmuon', 'production', 'mass', 'signal',
                 'iso', 'flight_dist_sig2', 'isolatione', 'isolationa', 'iso_min',
                 'isolationf']
    features_imp = list(f for f in train.columns if f not in filter_imp)
    ugbc_imp = UGradientBoostingClassifier(loss=loss, n_estimators=550,
                                 max_depth=6,
                                 learning_rate=0.15,
                                 train_features=features_imp,
                                 subsample=0.7,
                                 random_state=123)
    ugbc_imp.fit(train[features_imp + ['mass']], train['signal'])
    print("Importance preparation complete...")
if DO_IMP:
    imp = permutation_importances(ugbc_imp, train, train['signal'], features_imp)
    imp_df=pd.DataFrame(data=imp,index=features_imp).sort_values(0,ascending=False)
    print(imp_df.tail(10))
    roc_auc_truncated(train['signal'],ugbc_imp.predict_proba(train[features_imp])[:, 1])
    features = features_imp
# models and feature sets are assumed to be from prior gramolin run
if DO_GRAMOLIN_IMP:
    imp = perm_ens_imp([gramolin1,gramolin2], train['signal'], [list1,list2])
%time ugbc.fit(train_ugbc[features+['mass']], train_ugbc['signal'])
#--------------------  prediction ---------------------#
print ('----------------------------------------------')
print("Make predictions on the test set")
test_probs = ugbc.predict_proba(test_ugbc[features])[:,1]
if DO_5_ENS or DO_GRAMOLIN:
    test_predictions['ugbc_pred'] = test_probs
submission = pd.DataFrame({"id": test_ugbc["id"], "prediction": test_probs})
submission.to_csv("ugbc_features.csv", index=False)
print("UGBC Predictions done...")
if DO_5_ENS or DO_GRAMOLIN:
#    test_predictions['avg_preds'] = test_predictions.mean(axis=1)
    g_weight = .92      # gramolin weight
    u_weight = .08      # ugbc weight
    # test_predictions['prediction'] = add_noise(0.5*((g_weight*test_predictions['gramolin']) + (u_weight*test_predictions['ugbc_pred'])),.2)
    test_predictions['prediction'] = 0.5*((g_weight*test_predictions['gramolin']) + (u_weight*test_predictions['ugbc_pred']))
    test_predictions['id'] = test_ugbc['id']
    test_predictions[['id', 'prediction']].to_csv("ensembled.csv", index=False, header=["id", "prediction"])
    if (DO_5_ENS):
        g_weight = .49      # gramolin weight
        u_weight = .01      # ugbc weight
        f_weight = .50      # five lines weight
        # test_predictions['prediction'] = add_noise((g_weight*test_predictions['gramolin'] + (u_weight*test_predictions['ugbc_pred']) + f_weight*test_preds_1)/3)
        test_predictions['prediction'] = (g_weight*test_predictions['gramolin'] + (u_weight*test_predictions['ugbc_pred']) + f_weight*test_preds_1)/3
        test_predictions[['id', 'prediction']].to_csv("ensemble_all.csv", index=False, header=["id", "prediction"])
print("Ensemble Predictions done...")