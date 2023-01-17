import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from scipy import signal

from biosppy.signals import ecg, eeg, resp, eda



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
dtypes = {"crew": "int8",

          "experiment": "category",

          "time": "float32",

          "seat": "int8",

          "eeg_fp1": "float32",

          "eeg_f7": "float32",

          "eeg_f8": "float32",

          "eeg_t4": "float32",

          "eeg_t6": "float32",

          "eeg_t5": "float32",

          "eeg_t3": "float32",

          "eeg_fp2": "float32",

          "eeg_o1": "float32",

          "eeg_p3": "float32",

          "eeg_pz": "float32",

          "eeg_f3": "float32",

          "eeg_fz": "float32",

          "eeg_f4": "float32",

          "eeg_c4": "float32",

          "eeg_p4": "float32",

          "eeg_poz": "float32",

          "eeg_c3": "float32",

          "eeg_cz": "float32",

          "eeg_o2": "float32",

          "ecg": "float32",

          "r": "float32",

          "gsr": "float32",

          "event": "category",

         }
train_df = pd.read_csv('../input/train.csv', dtype=dtypes)

train_df.info()
test_df = pd.read_csv('../input/test.csv', dtype=dtypes)

test_df.info()
np.unique(train_df['crew'])
np.unique(test_df['crew'])
np.unique(train_df['experiment'])
np.unique(test_df['experiment'])
plt.figure(figsize=(15,10))

sns.countplot(train_df['event'])

plt.xlabel("State", fontsize=12)

plt.ylabel("Rows Count", fontsize=12)

plt.show()
subset = train_df.loc[(train_df['crew'] == 1) & (train_df['seat'] == 1) & (train_df['experiment'] == 'DA')]



subset = subset.sort_values(by='time')

events = subset['event'].values



event_dict = {

    'A': 0,

    'B': 1,

    'C': 2,

    'D': 3

}



events = list(map(lambda e:event_dict[e], events))



plt.plot(subset['ecg'].values)

plt.figure()

plt.plot(events)
ecg_out = ecg.ecg(signal=subset['ecg'].values, sampling_rate=256., show=False)

plt.plot(ecg_out['heart_rate_ts'], ecg_out['heart_rate'])
plt.plot(subset['r'].values)

plt.figure()

resp_out = resp.resp(signal=subset['r'].values, sampling_rate=256., show=False)

plt.plot(resp_out['resp_rate_ts'], resp_out['resp_rate'])
eeg_features = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2"]



eeg_out = eeg.get_power_features(signal=subset[eeg_features].values, sampling_rate=256.)

plt.plot(subset['eeg_fp1'].values)

plt.title('raw eeg fp1')

plt.figure()

plt.plot(eeg_out['ts'], eeg_out['theta'][:,0])

plt.title('theta eeg fp1')

plt.figure()

plt.plot(eeg_out['ts'], eeg_out['alpha_low'][:,0])

plt.title('alpha_low eeg fp1')

plt.figure()

plt.plot(eeg_out['ts'], eeg_out['alpha_high'][:,0])

plt.title('alpha_high eeg fp1')

plt.figure()

plt.plot(eeg_out['ts'], eeg_out['beta'][:,0])

plt.title('beta eeg fp1')

plt.figure()

plt.plot(eeg_out['ts'], eeg_out['gamma'][:,0])

plt.title('gamma eeg fp1')
plt.plot(subset['gsr'].values)

plt.title('raw gsr')

plt.figure()

gsr_out = eda.eda(signal=subset['gsr'].values, sampling_rate=256., show=False, min_amplitude=0)

plt.plot(subset['time'].values[gsr_out['onsets']], gsr_out['amplitudes'])

plt.title('gsr amplitude')
from scipy.interpolate import interp1d



def map_timestamped_feature_to_data(df_times, new_feature_ts, new_feature_data):

    f = interp1d(new_feature_ts, new_feature_data, kind='cubic', fill_value="extrapolate")

    return f(df_times)
eeg_features = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2"]



def extract_features_for_pilot(data_df, pilot_loc):

    curr_pilot_values = data_df.loc[pilot_loc][['time', 'ecg', 'r', 'gsr'] + eeg_features].values

    curr_pilot_values = curr_pilot_values[curr_pilot_values[:,0].argsort()]

    

    if np.allclose(curr_pilot_values[:,1], 0, rtol=1e-10):

        data_df.loc[pilot_loc, 'ecg'] = np.nan

        print(f'\tmissing egc')

    else:

        try:

            heart_out = ecg.ecg(signal=curr_pilot_values[:,1], sampling_rate=256., show=False)

            heart_rate = heart_out['heart_rate']

            heart_rate_ts = heart_out['heart_rate_ts']

            data_df.loc[pilot_loc, 'heart_rate'] = map_timestamped_feature_to_data(curr_pilot_values[:,0], heart_rate_ts, heart_rate)

        except ValueError:

            print(f'\tfailed "heart_rate" extraction')

    

    if np.allclose(curr_pilot_values[:,2], 0, rtol=1e-10):

        data_df.loc[pilot_loc, 'r'] = np.nan

        print(f'\tmissing r')

    else:

        try:

            resp_out = resp.resp(signal=curr_pilot_values[:,2], sampling_rate=256., show=False)

            resp_rate = resp_out['resp_rate']

            resp_rate_ts = resp_out['resp_rate_ts']

            data_df.loc[pilot_loc, 'resp_rate'] = map_timestamped_feature_to_data(curr_pilot_values[:,0], resp_rate_ts, resp_rate)

        except ValueError:

            print(f'\tfailed "resp_rate" extraction')

        

    if np.allclose(curr_pilot_values[:,3], 0, rtol=1e-10):

        data_df.loc[pilot_loc, 'gsr'] = np.nan

        print(f'\tmissing gsr')

    else:

        try:

            gsr_out = eda.eda(signal=curr_pilot_values[:,3], sampling_rate=256., show=False)

            gsr_amp = gsr_out['amplitudes']

            gsr_amp_ts = curr_pilot_values[gsr_out['onsets'], 0]

            data_df.loc[pilot_loc, 'gsr_amp'] = map_timestamped_feature_to_data(curr_pilot_values[:,0], gsr_amp_ts, gsr_amp)

        except IndexError:

            print(f'\tfailed "gsr_amp" extraction')

        except ValueError:

            print(f'\tfailed "gsr_amp" extraction')

        

    try:

        eeg_feat_out = eeg.get_power_features(signal=curr_pilot_values[:,4:], sampling_rate=256.)

        eeg_ts = eeg_feat_out['ts']

        eeg_theta = eeg_feat_out['theta']

        eeg_alpha_low = eeg_feat_out['alpha_low']

        eeg_alpha_high = eeg_feat_out['alpha_high']

        eeg_beta = eeg_feat_out['beta']

        eeg_gamma = eeg_feat_out['gamma']

        for i, eeg_feature in enumerate(eeg_features):

            data_df.loc[pilot_loc, eeg_feature + '_theta'] = map_timestamped_feature_to_data(curr_pilot_values[:,0], eeg_ts, eeg_theta[:,i])

            data_df.loc[pilot_loc, eeg_feature + '_alpha_low'] = map_timestamped_feature_to_data(curr_pilot_values[:,0], eeg_ts, eeg_alpha_low[:,i])

            data_df.loc[pilot_loc, eeg_feature + '_alpha_high'] = map_timestamped_feature_to_data(curr_pilot_values[:,0], eeg_ts, eeg_alpha_high[:,i])

            data_df.loc[pilot_loc, eeg_feature + '_beta'] = map_timestamped_feature_to_data(curr_pilot_values[:,0], eeg_ts, eeg_beta[:,i])

            data_df.loc[pilot_loc, eeg_feature + '_gamma'] = map_timestamped_feature_to_data(curr_pilot_values[:,0], eeg_ts, eeg_gamma[:,i])

    except ValueError:

        print(f'\tfailed "eeg"')

enhanced_train_df = train_df.copy()

enhanced_train_df['heart_rate'] = np.nan

enhanced_train_df['resp_rate'] = np.nan

enhanced_train_df['gsr_amp'] = np.nan



for eeg_feature in eeg_features:

    enhanced_train_df[eeg_feature + '_theta'] = np.nan

    enhanced_train_df[eeg_feature + '_alpha_low'] = np.nan

    enhanced_train_df[eeg_feature + '_alpha_high'] = np.nan

    enhanced_train_df[eeg_feature + '_beta'] = np.nan

    enhanced_train_df[eeg_feature + '_gamma'] = np.nan



for tup in [(c,s,e) for c in np.unique(enhanced_train_df['crew']) for s in np.unique(enhanced_train_df['seat']) for e in np.unique(enhanced_train_df['experiment'])]:

    c, s, e = tup

    pilot_loc = (enhanced_train_df['crew'] == c) & (enhanced_train_df['seat'] == s) & (enhanced_train_df['experiment'] == e)

    print(f'extracting for {tup}')

    extract_features_for_pilot(enhanced_train_df, pilot_loc)

    
features = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2", "ecg", "r", "gsr", 'heart_rate', 'resp_rate', 'gsr_amp']



for eeg_feature in eeg_features:

    features += [eeg_feature + '_theta']

    features += [eeg_feature + '_alpha_low']

    features += [eeg_feature + '_alpha_high']

    features += [eeg_feature + '_beta']

    features += [eeg_feature + '_gamma']



subset = enhanced_train_df.loc[(enhanced_train_df['crew'] == 5) & (enhanced_train_df['seat'] == 0)]



f, ax = plt.subplots(figsize=(10, 8))

corr = subset[features].corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)
params = {

    "objective" : "multiclass",

    "num_class": 4,

    "metric" : "multi_error",

    "num_leaves" : 30,

    "max_depth" : 8,

    "min_child_weight" : 50,

    "learning_rate" : 0.1,

    "bagging_fraction" : 0.7,

    "feature_fraction" : 0.7,

    "bagging_seed" : 42,

    "verbosity" : -1,

    "n_estimators": 25

}



features = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2", "ecg", "r", "gsr", 'heart_rate', 'resp_rate', 'gsr_amp']



for eeg_feature in eeg_features:

    features += [eeg_feature + '_theta']

    features += [eeg_feature + '_alpha_low']

    features += [eeg_feature + '_alpha_high']

    features += [eeg_feature + '_beta']

    features += [eeg_feature + '_gamma']
from sklearn import preprocessing as prep



le = prep.LabelEncoder()

le.fit(enhanced_train_df['event'])
from sklearn.model_selection import train_test_split

# import xgboost as xgb

import lightgbm as lgb

from sklearn import metrics



conf_matrix = np.zeros((4,4))



for tup in [(x,y) for x in np.unique(enhanced_train_df['crew']) for y in np.unique(enhanced_train_df['seat'])]:



    one_pilot_train_df = enhanced_train_df[(enhanced_train_df['crew'] == tup[0]) & (enhanced_train_df['seat'] == tup[1])]

    temp_train_df, temp_val_df = train_test_split(one_pilot_train_df, test_size=0.4, random_state=42)



    X_train = temp_train_df[features]

    y_train = le.transform(temp_train_df['event'])



    X_test = temp_val_df[features]

    y_test = le.transform(temp_val_df['event'])

    

    clf = lgb.LGBMClassifier(**params)

    clf.fit(X_train, y_train)

    results = clf.predict_proba(X_test)

    

    y_pred = np.argmax(results, 1)

    

    conf_matrix += metrics.confusion_matrix(y_test, y_pred)



    print(f'pilot {tup} logloss: {metrics.log_loss(y_test, results, eps=1e-15)}')

import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()



f = plt.figure()

f.set_figwidth(12)

plot_confusion_matrix(conf_matrix, le.classes_, normalize=True)
one_pilot_train_df = enhanced_train_df[(enhanced_train_df['crew'] == 5) & (enhanced_train_df['seat'] == 0)]

X_train = one_pilot_train_df[features]

y_train = le.transform(one_pilot_train_df['event'])



clf = lgb.LGBMClassifier(**params)

clf.fit(X_train, y_train)



lgb.plot_importance(clf, max_num_features=30, figsize=(20,10))
all_results = np.zeros((test_df['id'].shape[0], 5))

all_results[:, 0] = test_df['id']



for tup in [(x,y) for x in np.unique(test_df['crew']) for y in np.unique(test_df['seat'])]:

    

    print(f'pilot {tup}')

    

    train_rows = (enhanced_train_df['crew'] == tup[0]) & (enhanced_train_df['seat'] == tup[1])

    test_rows = (test_df['crew'] == tup[0]) & (test_df['seat'] == tup[1])

    

    one_pilot_train_df = enhanced_train_df[train_rows]

    one_pilot_test_df = test_df[test_rows].copy()

    extract_features_for_pilot(one_pilot_test_df, one_pilot_test_df.index.values)

    

    for f in features:

        if f not in one_pilot_test_df.columns:

            one_pilot_test_df[f] = np.nan

    

    X_train = one_pilot_train_df[features]

    y_train = le.transform(one_pilot_train_df['event'])

    

    X_test = one_pilot_test_df[features]

    

    clf = lgb.LGBMClassifier(**params)

    clf.fit(X_train, y_train)

    pilot_results = clf.predict_proba(X_test)

    

    all_results[np.where(test_rows), 1:5] = pilot_results
submission = pd.DataFrame(all_results, columns=['id', 'A', 'B', 'C', 'D'])

submission['id'] = submission['id'].astype(int)



submission.head()
submission.to_csv("submission.csv", index=False)