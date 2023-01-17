import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
import random
train_file = "../input/train.csv"
train = pd.read_csv(train_file)
test_file = "../input/test.csv"
test = pd.read_csv(test_file)
def marshall_palmer(ref, minutes_past):
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in xrange(1, len(minutes_past)):
        valid_time[n] = int(minutes_past.iloc[n]) - int(minutes_past.iloc[n-1])
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for dbz, hours in zip(ref, valid_time):
        if np.isfinite(dbz):
            mmperhr = pow(pow(10, dbz/10)/200, 0.625)
            sum = sum + mmperhr * hours
    return sum
def applymp(hour):
    hour = hour.sort('minutes_past', ascending=True)
    est = marshall_palmer(hour['Ref'], hour['minutes_past'])
    return est
mptrain = train.groupby(train.index).apply(applymp)
mptest = test.groupby(test.index).apply(applymp)
train['RefSum'] = train['Ref']
test['RefSum'] = test['Ref']
train['KdpSum'] = train['Kdp']
test['KdpSum'] = test['Kdp']
def groupdf(df, train=True):
    grouped = df.groupby('Id')
    if train == True:
        collapsed = grouped.agg({
            'Id'           : np.mean,
            'minutes_past' : len,
            'radardist_km' : np.mean,
            'RefSum'       : np.sum,
            'Ref'          : np.mean,
            'Ref_5x5_10th' : np.mean,
            'Ref_5x5_50th' : np.mean,
            'Ref_5x5_90th' : np.mean,
            'RefComposite' : np.mean,
            'RefComposite_5x5_10th' : np.mean,
            'RefComposite_5x5_50th' : np.mean,
            'RefComposite_5x5_90th' : np.mean,
            'Zdr'          : np.mean,
            'Zdr_5x5_10th' : np.mean,
            'Zdr_5x5_50th' : np.mean,
            'Zdr_5x5_90th' : np.mean,
            'Kdp'          : np.mean,
            'KdpSum'       : np.sum,
            'Kdp_5x5_10th' : np.mean,
            'Kdp_5x5_50th' : np.mean,
            'Kdp_5x5_90th' : np.mean,
            'RhoHV'        : np.mean,
            'Expected'     : np.mean, 
            });
    else:
        collapsed = grouped.agg({
            'Id'           : np.mean,
            'minutes_past' : len,
            'radardist_km' : np.mean,
            'RefSum'       : np.sum,
            'Ref'          : np.mean,
            'Ref_5x5_10th' : np.mean,
            'Ref_5x5_50th' : np.mean,
            'Ref_5x5_90th' : np.mean,
            'RefComposite' : np.mean,
            'RefComposite_5x5_10th' : np.mean,
            'RefComposite_5x5_50th' : np.mean,
            'RefComposite_5x5_90th' : np.mean,
            'Zdr'          : np.mean,
            'Zdr_5x5_10th' : np.mean,
            'Zdr_5x5_50th' : np.mean,
            'Zdr_5x5_90th' : np.mean,
            'Kdp'          : np.mean,
            'KdpSum'       : np.sum,
            'Kdp_5x5_10th' : np.mean,
            'Kdp_5x5_50th' : np.mean,
            'Kdp_5x5_90th' : np.mean,
            'RhoHV'        : np.mean
            });
    return collapsed
train_collapsed = groupdf(train, True)
test_collapsed = groupdf(test, False)
train_collapsed['mp'] = mptrain
test_collapsed['mp'] = mptest
train_collapsed['rlog'] = np.log1p(train_collapsed['Expected'])
train_collapsed, cv_collapsed = train_test_split(train_collapsed, test_size = 0.15)
train_collapsed.to_csv('..input/train-collapsed-all-features.csv')
cv_collapsed.to_csv('..input/cv-collapsed-all-features.csv')
test_collapsed.to_csv('..input/test-collapsed-all-features.csv')