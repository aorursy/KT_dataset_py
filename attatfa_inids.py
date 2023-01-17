import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import seaborn as sns

from imblearn.under_sampling import RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE

import time

import joblib
import glob



all_files = glob.glob("../input/cicids2017/MachineLearningCSV/MachineLearningCVE/*.csv")

df_from_each_file = (pd.read_csv(f) for f in all_files)

concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

concatenated_df.to_csv('full.csv')

#file = "../input/cicids2017/MachineLearningCSV/MachineLearningCVE/*.csv"

file = "./full.csv"
df = pd.read_csv(file)

df = df.reset_index()
df.head()
df.columns = df.columns.str.strip()

df.columns = df.columns.str.replace(' ', '_')

df.columns = map(str.lower, df.columns)
df = df.sample(frac=1).reset_index(drop=True)
null_columns=df.columns[df.isnull().any()]

df[null_columns].isnull().sum() 
df.head()
df.rename(columns={'fwd_avg_packets/bulk':'fwd_packet/bulk_avg', 'bwd_avg_bulk_rate':'bwd_bulk_rate_avg','fwd_avg_bulk_rate':'fwd_bulk_rate_avg', 'bwd_avg_packets/bulk':'bwd_packet/bulk_avg', 'fwd_avg_bytes/bulk':'fwd_bytes/bulk_avg', 'avg_bwd_segment_size':'bwd_segment_size_avg', 'avg_fwd_segment_size':'fwd_segment_size_avg','cwe_flag_count':'cwr_flag_count','total_length_of_bwd_packets':'total_length_of_bwd_packet','total_length_of_fwd_packets': 'total_length_of_fwd_packet','total_fwd_packets': 'total_fwd_packet','total_backward_packets': 'total_bwd_packets', 'init_win_bytes_forward': 'fwd_init_win_bytes', 'init_win_bytes_backward':'bwd_init_win_bytes', 'act_data_pkt_fwd':'fwd_act_data_pkts', 'min_seg_size_forward':'fwd_seg_size_min'}, inplace=True)
df['label'].value_counts()
#df['Label'] = df['Label'].replace(r'PortScan|Dos Hulk|DDos*|Web*|', 1)

df['label'] = df['label'].replace('DDoS', 1)

df['label'] = df['label'].replace('DoS Hulk', 1)

df['label'] = df['label'].replace('PortScan', 1)

df['label'] = df['label'].replace('DoS GoldenEye', 1)

df['label'] = df['label'].replace('DoS slowloris', 1)

df['label'] = df['label'].replace('DoS Slowhttptest', 1)

df['label'] = df['label'].replace('Web Attack � Brute Force', 1)

df['label'] = df['label'].replace('Web Attack � XSS', 1)

df['label'] = df['label'].replace('Infiltration', 1)

df['label'] = df['label'].replace('Web Attack � Sql Injection', 1)

df['label'] = df['label'].replace('Heartbleed', 1)

df['label'] = df['label'].replace('FTP-Patator', 1)

df['label'] = df['label'].replace('SSH-Patator', 1)

df['label'] = df['label'].replace('Bot', 1)

df['label'] = df['label'].replace('BENIGN', 0)
df['label'].value_counts()
#np.any(np.isnan(df))

df = df.dropna()

df['label'].value_counts()
col_mask=df.isnull().any(axis=0)

row_mask=df.isnull().any(axis=1)

df.loc[row_mask,col_mask]
df.shape
df = df.dropna()
df = df.apply (pd.to_numeric, errors='coerce')

df = df.dropna()

df = df.reset_index(drop=True)
df.shape
col_mask=df.isnull().any(axis=0)

row_mask=df.isnull().any(axis=1)

df.loc[row_mask,col_mask]
#np.any(np.isnan(df))
#np.all(np.isfinite(df))
def clean_dataset(df):

    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

    df.dropna(inplace=True)

    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)

    return df[indices_to_keep].astype(np.float64)
df = clean_dataset(df)
df.shape
null_columns=df.columns[df.isnull().any()]

df[null_columns].isnull().sum() 
np.all(np.isfinite(df))
df.describe()
X = df.drop(['index','unnamed:_0','label','destination_port', 'min_packet_length', 'max_packet_length', 'fwd_header_length.1','bwd_avg_bytes/bulk'], axis=1)

y = df['label']
down_dataset = {

    0: 600000,

    1: 556556

}

down_df=RandomUnderSampler(sampling_strategy=down_dataset, random_state=0) 
from imblearn import under_sampling



rus = under_sampling.RandomUnderSampler(sampling_strategy=down_dataset)
test_percentage = 0.25

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage)
y_train.value_counts()
X, y = down_df.fit_sample(X, y) 
y.value_counts()
X.loc[:, (X == 0).all()].describe()
drop_list = ["bwd_psh_flags", "bwd_urg_flags", "fwd_bytes/bulk_avg", "fwd_packet/bulk_avg", "fwd_bulk_rate_avg", "bwd_packet/bulk_avg", "bwd_bulk_rate_avg"]

len(drop_list)
X = X.drop(drop_list, axis=1)
X.shape
from sklearn.ensemble.forest import RandomForestClassifier

print("Total dataset: {}".format(X.shape))

print("Training dataset: {}:".format(X_train.shape))

print("Testing dataset: {}:".format(X_test.shape))
X.shape
rf = RandomForestClassifier(random_state=0, n_jobs=-1)

rfModel = rf.fit(X,y)

importance = rfModel.feature_importances_
sorted(zip(map(lambda x: round(x, 4), rfModel.feature_importances_)), 

             reverse=True)
for i,v in enumerate(importance):

	print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance

pyplot.bar([x for x in range(len(importance))], importance)

pyplot.show()
important
print("Total dataset: {}".format(df.shape))

print("Training dataset: {}:".format(X_train.shape))

print("Testing dataset: {}:".format(X_test.shape))
print("------------------ LogisticRegression -----------------")

start = time.time()
clf_lr = LogisticRegression( solver='newton-cg')

clf_lr.fit(X_train, y_train)
print("Total time take {}".format(time.time() - start))
##Evaluating the model
y_pred = clf_lr.predict(X_test)

print("Model accuracy on test dataset")

clf_lr.score(X_test, y_test)
tn, fp, fn, tp  = metrics.confusion_matrix(y_test, y_pred, labels=None, sample_weight=None).ravel()
print('true positives  rate {}'.format(tp))

print('false positives  rate {}'.format(fp))

print('true negatives  rate {}'.format(tn))

print('false negatives  rate {}'.format(fn))

print("F1 Score = {}".format(metrics.f1_score(y_test, y_pred)))

print("Recall {}".format(tp / (tp + fn)))

print("Precession {}".format(tp / (tp + fp)))
file_name = "LogisticRegression.sav"

joblib.dump(clf_lr, file_name)
from sklearn.ensemble import RandomForestClassifier
print('------- RandomForest------------------\n')
start = time.time()
clf_rf = RandomForestClassifier(max_depth=2, random_state=0)

clf_rf.fit(X_train, y_train)
print("Total time take for {}".format(time.time() - start))
y_pred = clf_rf.predict(X_test)

print("Model accuracy on test dataset")

clf_rf.score(X_test, y_test)
tn, fp, fn, tp  = metrics.confusion_matrix(y_test, y_pred, labels=None, sample_weight=None).ravel()
print('true positives  rate {}'.format(tp))

print('false positives  rate {}'.format(fp))

print('true negatives  rate {}'.format(tn))

print('false negatives  rate {}'.format(fn))

print("F1 Score = {}".format(metrics.f1_score(y_test, y_pred)))

print("Recall {}".format(tp / (tp + fn)))

print("Precession {}".format(tp / (tp + fp)))
file_name = "RF.sav"

joblib.dump(clf_rf, file_name)
from sklearn.naive_bayes import GaussianNB
print('------- Naive_Bayes------------------\n')
start = time.time()
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train)
print("Total time take for RF {}".format(time.time() - start))
y_pred = gnb.predict(X_test)

print("Model accuracy on test dataset")

gnb.score(X_test, y_test)
tn, fp, fn, tp  = metrics.confusion_matrix(y_test, y_pred, labels=None, sample_weight=None).ravel()
print('true positives  rate {}'.format(tp))

print('false positives  rate {}'.format(fp))

print('true negatives  rate {}'.format(tn))

print('false negatives  rate {}'.format(fn))

print("F1 Score = {}".format(metrics.f1_score(y_test, y_pred)))

print("Recall {}".format(tp / (tp + fn)))

print("Precession {}".format(tp / (tp + fp)))
file_name = "gnb.sav"

joblib.dump(gnb, file_name)