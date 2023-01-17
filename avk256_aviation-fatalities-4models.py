# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/reducing-commercial-aviation-fatalities/train.csv")

test_df = pd.read_csv("/kaggle/input/reducing-commercial-aviation-fatalities/test.csv")

sample_sub = pd.read_csv("/kaggle/input/reducing-commercial-aviation-fatalities/sample_submission.csv")
sample_sub.sample(5)
train_df.info()
train_df.sample(10)
test_df.sample(10)
train_df.describe()
from matplotlib import pyplot as plt

import seaborn as sns



plt.figure(figsize=(15,10))

sns.countplot(train_df['event'])

plt.xlabel("State of the pilot", fontsize=12)

plt.ylabel("Count", fontsize=12)

plt.title("Target repartition", fontsize=15)

plt.show()
plt.figure(figsize=(15,10))

sns.countplot('experiment', hue='event', data=train_df)

plt.xlabel("Experiment and state of the pilot", fontsize=12)

plt.ylabel("Count (log)", fontsize=12)

plt.yscale('log')

plt.title("Target repartition for different experiments", fontsize=15)

plt.show()
eeg_features = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2"]
plt.figure(figsize=(20,25))

i = 0



for egg in eeg_features:

    i += 1

    plt.subplot(5, 4, i)

    sns.boxplot(x='event', y=egg, data=train_df.sample(50000), showfliers=False)



plt.show()
plt.figure(figsize=(20,25))

plt.title('Eeg features distributions')

i = 0



for eeg in eeg_features:

    i += 1

    plt.subplot(5, 4, i)

    sns.distplot(test_df.sample(10000)[eeg], label='Test set', hist=False)

    sns.distplot(train_df.sample(10000)[eeg], label='Train set', hist=False)

    plt.xlim((-500, 500))

    plt.legend()

    plt.xlabel(eeg, fontsize=12)



plt.show()
plt.figure(figsize=(15,10))

sns.distplot(test_df['ecg'], label='Test set')

sns.distplot(train_df['ecg'], label='Train set')

plt.legend()

plt.xlabel("Electrocardiogram Signal (µV)", fontsize=12)

plt.title("Electrocardiogram Signal Distribution", fontsize=15)

plt.show()
plt.figure(figsize=(15,10))

sns.distplot(test_df['r'], label='Test set')

sns.distplot(train_df['r'], label='Train set')

plt.legend()

plt.xlabel("Respiration Signal (µV)", fontsize=12)

plt.title("Respiration Signal Distribution", fontsize=15)

plt.show()
plt.figure(figsize=(15,10))

sns.distplot(test_df['gsr'], label='Test set')

sns.distplot(train_df['gsr'], label='Train set')

plt.legend()

plt.xlabel("Electrodermal activity measure (µV)", fontsize=12)

plt.title("Electrodermal activity Distribution", fontsize=15)

plt.show()
features_n = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2", "ecg", "r", "gsr"]
train_df['pilot'] = 100 * train_df['seat'] + train_df['crew']

test_df['pilot'] = 100 * test_df['seat'] + test_df['crew']

print("Number of pilots : ", len(train_df['pilot'].unique()))
train_df.sample(10)
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler

def normalize_by_pilots(df):

    pilots = df["pilot"].unique()

    for pilot in tqdm(pilots):

        ids = df[df["pilot"] == pilot].index

        scaler = MinMaxScaler()

        df.loc[ids, features_n] = scaler.fit_transform(df.loc[ids, features_n])

        

    return df
train_df = normalize_by_pilots(train_df)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in split.split(train_df,train_df["event"]):

    strat_train = train_df.loc[train_index]

    strat_test = train_df.loc[test_index]

plt.figure(figsize=(15,10))

sns.countplot(strat_train['event'],order=['A','B','C','D'])

plt.xlabel("State of the pilot", fontsize=12)

plt.ylabel("Count", fontsize=12)

plt.title("Target repartition. Train", fontsize=15)

plt.show()
plt.figure(figsize=(15,10))

sns.countplot(strat_test['event'],order=['A','B','C','D'])

plt.xlabel("State of the pilot", fontsize=12)

plt.ylabel("Count", fontsize=12)

plt.title("Target repartition. Test", fontsize=15)

plt.show()
print(f"Training on {strat_train.shape[0]} samples.")

print(f"Testing on {strat_test.shape[0]} samples.")
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

import time
dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

strat_train["event"] = strat_train["event"].apply(lambda x: dic[x])

strat_test["event"] = strat_test["event"].apply(lambda x: dic[x])

x_train = strat_train[features_n]

y_train = strat_train['event']

x_test = strat_test[features_n]

y_test = strat_test['event']
start = time.time()

clf_1 = DecisionTreeClassifier(max_depth=30)

clf_1 = clf_1.fit(x_train, y_train)

end = time.time()

y_pred = clf_1.predict(x_test)

dec_tree_time = end - start
dec_tree_score = clf_1.score(x_test, y_test)

pr_score = precision_score(y_test, y_pred, average='weighted')

rc_score = recall_score(y_test, y_pred, average='weighted')

f1_score = f1_score(y_test, y_pred, average='weighted')

print('dec_tree_score ',dec_tree_score)

print('precision_score', pr_score)

print('recall_score', rc_score)

print('f1_score', f1_score)

print('dec_tree_time (s)', dec_tree_time)
cm = confusion_matrix(y_test, y_pred)

print('confusion_matrix\n',cm)

plt.matshow(cm)

plt.show()
start = time.time()

clf_2 = RandomForestClassifier(n_estimators=5, max_depth=20, random_state=0)

clf_2 = clf_2.fit(x_train, y_train)

end = time.time()

y_pred = clf_2.predict(x_test)

RF_time = end - start
print(y_pred)
RF_score = clf_2.score(x_test, y_test)

pr_score = precision_score(y_test, y_pred, average='weighted')

rc_score = recall_score(y_test, y_pred, average='weighted')

#f1_score = f1_score(list(y_test), y_pred)

print('RF_score ',RF_score)

print('precision_score', pr_score)

print('recall_score', rc_score)

# print('f1_score', f1_score)

print('RF_time (s)', RF_time)
cm = confusion_matrix(y_test, y_pred)

print('confusion_matrix\n',cm)

plt.matshow(cm)

plt.show()
# from sklearn.multiclass import OneVsRestClassifier

# from sklearn.ensemble import BaggingClassifier

# import time

# n_estimators = 3

# start = time.time()

# clf_3 = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))

# # clf_3 = OneVsRestClassifier(SVC(kernel='linear', probability=True))

# # clf_3 = SVC(gamma='auto')

# clf_3 = clf_3.fit(x_train, y_train)

# end = time.time()

# y_pred = clf_3.predict(x_test)

# SVC_time = end - start
# SVC_score = clf_3.score(x_test, y_test)

# pr_score = precision_score(y_test, y_pred, average='weighted')

# rc_score = recall_score(y_test, y_pred, average='weighted')

# f1_score = f1_score(y_test, y_pred, average='weighted')

# print('SVC_score ', SVC_score)

# print('precision_score', pr_score)

# print('recall_score', rc_score)

# print('f1_score', f1_score)
# cm = confusion_matrix(y_test, y_pred)

# print('confusion_matrix\n',cm)

# plt.matshow(cm)

# plt.show()
# start = time.time()

# clf_4 = KNeighborsClassifier(n_neighbors=3)

# clf_4 = clf_4.fit(x_train, y_train)

# end = time.time()

# y_pred = clf_4.predict(x_test)

# KNN_time = end - start
# KNN_score = clf_4.score(x_test, y_test)

# pr_score = precision_score(y_test, y_pred, average='weighted')

# rc_score = recall_score(y_test, y_pred, average='weighted')

# # f1_score = f1_score(y_test, y_pred, average='weighted')

# print('KNN_score ',KNN_score)

# print('precision_score', pr_score)

# print('recall_score', rc_score)

# # print('f1_score', f1_score)

# print('KNN_time', KNN_time)
# cm = confusion_matrix(y_test, y_pred)

# print('confusion_matrix\n',cm)

# plt.matshow(cm)

# plt.show()
start = time.time()

clf_5 = MLPClassifier(hidden_layer_sizes=(100), max_iter=10, alpha=0.0001,

                      solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

clf_5 = clf_5.fit(x_train, y_train)

end = time.time()

y_pred = clf_5.predict(x_test)

MLP_time = end - start
MLP_score = clf_5.score(x_test, y_test)

pr_score = precision_score(y_test, y_pred, average='weighted')

rc_score = recall_score(y_test, y_pred, average='weighted')

# f1_score = f1_score(y_test, y_pred, average='weighted')

print('MLP_score ',MLP_score)

print('precision_score', pr_score)

print('recall_score', rc_score)

# print('f1_score', f1_score)

print('MLP_time (s)', MLP_time)
cm = confusion_matrix(y_test, y_pred)

print('confusion_matrix\n',cm)

plt.matshow(cm)

plt.show()