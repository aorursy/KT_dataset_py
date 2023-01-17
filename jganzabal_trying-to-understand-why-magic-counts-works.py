import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from sklearn.metrics import roc_auc_score

import tqdm

from sklearn.linear_model import LogisticRegression
# GET INDICIES OF REAL TEST DATA FOR FE

#######################

# TAKE FROM YAG320'S KERNEL

# https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split



test_path = '../input/test.csv'

train_path = '../input/train.csv'



df_test = pd.read_csv(test_path)

df_test.drop(['ID_code'], axis=1, inplace=True)

df_test = df_test.values



unique_samples = []

unique_count = np.zeros_like(df_test)

for feature in range(df_test.shape[1]):

    _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)

    unique_count[index_[count_ == 1], feature] += 1



# Samples which have unique values are real the others are fake

real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]

synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]



print('Found',len(real_samples_indexes),'real test')

print('Found',len(synthetic_samples_indexes),'fake test')



###################



d = {}

for i in range(200): d['var_'+str(i)] = 'float32'

d['target'] = 'uint8'

d['ID_code'] = 'object'



train = pd.read_csv('../input/train.csv', dtype=d)

test = pd.read_csv('../input/test.csv', dtype=d)



print('Loaded',len(train),'rows of train')

print('Loaded',len(test),'rows of test')

print('Found',len(real_samples_indexes),'real test')

print('Found',len(synthetic_samples_indexes),'fake test')



###################



d = {}

for i in range(200): d['var_'+str(i)] = 'float32'

d['target'] = 'uint8'

d['ID_code'] = 'object'



train = pd.read_csv(train_path, dtype=d)

test = pd.read_csv(test_path, dtype=d)



print('Loaded',len(train),'rows of train')

print('Loaded',len(test),'rows of test')
# FREQUENCY ENCODE

def encode_FE(df,col,test):

    cv = df[col].value_counts()

    nm = col+'_FE'

    df[nm] = df[col].map(cv)

    test[nm] = test[col].map(cv)

    test[nm].fillna(0,inplace=True)

    if cv.max()<=255:

        df[nm] = df[nm].astype('uint8')

        test[nm] = test[nm].astype('uint8')

    else:

        df[nm] = df[nm].astype('uint16')

        test[nm] = test[nm].astype('uint16')        

    return



test['target'] = -1

comb = pd.concat([train,test.loc[real_samples_indexes]],axis=0,sort=True)

for i in range(200): 

    encode_FE(comb,'var_'+str(i),test)

train = comb[:len(train)]; del comb

print('Added 200 new magic features!')
df_train_data = train.drop(columns=['ID_code'])

df_test_data = test.drop(columns=['ID_code'])
import numpy as np

import scipy.interpolate as interpolate



def inverse_transform_sampling(data, n_bins, n_samples, draw_hist=False):

    # This function returns samples with the same distribution of data

    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)

    cum_values = np.zeros(bin_edges.shape)

    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))

    inv_cdf = interpolate.interp1d(cum_values, bin_edges)

    r = np.random.rand(n_samples)

    samples = inv_cdf(r)

    if draw_hist:

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,4))

        ax1.hist(data, n_bins, density=True)

        ax1.set_title('Original Data Hist')

        ax2.hist(samples, n_bins, density=True)

        ax2.set_title('Sampled Data Hist')

        plt.show()

    return samples
var_i = 82

data_0 = df_train_data[df_train_data['target']==0][f'var_{var_i}'].values

data_1 = df_train_data[df_train_data['target']==1][f'var_{var_i}'].values

print(f'For target 0 and var_{var_i}')

samples_0 = inverse_transform_sampling(data_0, 100, len(data_0), draw_hist=True)

print(f'For target 1 and var_{var_i}')

samples_1 = inverse_transform_sampling(data_1, 50, len(data_1), draw_hist=True)

count_1 = len(set(np.round(data_1, 4)))

count_sample_1 = len(set(np.round(samples_1, 4)))

count_0 = len(set(np.round(data_0, 4)))

count_sample_0 = len(set(np.array(samples_0*10000, dtype=int)))

print(f'Target 1: Unique original data count({count_1}) vs Unique sampled data({count_sample_1}): {count_1/count_sample_1}')

print(f'Target 0: Unique original data count({count_0}) vs Unique sampled data({count_sample_0}): {count_0/count_sample_0}')
counts_1 = []

counts_0 = []

counts_sample_1 = []

counts_sample_0 = []

for var_i in tqdm.tqdm(range(200)):

    data_0 = df_train_data[df_train_data['target']==0][f'var_{var_i}'].values

    data_1 = df_train_data[df_train_data['target']==1][f'var_{var_i}'].values

    samples_0 = inverse_transform_sampling(data_0, 100, len(data_0))

    samples_1 = inverse_transform_sampling(data_1, 50, len(data_1))

    count_1 = len(set(np.round(data_1, 4)))

    count_sample_1 = len(set(np.round(samples_1, 4)))

    count_0 = len(set(np.round(data_0, 4)))

    count_sample_0 = len(set(np.round(samples_0, 4)))

    counts_1.append(count_1)

    counts_0.append(count_0)

    counts_sample_1.append(count_sample_1)

    counts_sample_0.append(count_sample_0)
ones_quotient = np.array(counts_1)/np.array(counts_sample_1)

print(ones_quotient.mean(), ones_quotient.std())

zeros_quotient = np.array(counts_0)/np.array(counts_sample_0)

print(zeros_quotient.mean(), zeros_quotient.std())
plt.plot(ones_quotient)

plt.plot(zeros_quotient)

plt.show()
plt.hist(zeros_quotient, 20)

plt.hist(ones_quotient, 20)

plt.show()
def create_dataset(append_counts, decimals_ones=4, decimals_zeros=4,  N_ones = 20_000, N_zeros = 180_000, mean = 0, std = 3):

    # Mean and std could be changed. The selected where to demostrate the effect



    # Sample normal distribution variable and round it with decimals_ones decimals

    mult = np.power(10, decimals_ones)

    normal_x_ones = np.array(mult*np.random.normal(mean, std, (N_ones,1)), dtype=int)/mult

    # Append ones

    data_ones = np.append(normal_x_ones, np.ones((N_ones,1)), axis=1)



    # Sample normal distribution variable and round it with decimals_zeros decimals

    mult = np.power(10, decimals_zeros)

    normal_x_zeros = np.array(mult*np.random.normal(mean, std, (N_zeros,1)), dtype=int)/mult

    # Append zeros

    data_zeros = np.append(normal_x_zeros, np.zeros((N_zeros,1)), axis=1)



    # Append zeros with ones

    data = np.append(data_zeros, data_ones, axis=0)

    X_train = data[:,0].reshape(-1,1)

    y_train = data[:,1]



    if append_counts:

        # Append counts

        values, indexes, inv, count = np.unique(X_train, return_index=True, return_inverse=True, return_counts=True)

        count_data = count[inv].reshape(-1,1)



        X_train = np.append(X_train, count_data, axis=1)

        

    return X_train, y_train
X_train, y_train = create_dataset(append_counts=False, decimals_ones=4, decimals_zeros=4)

print('X_train:\n', X_train)

print('y_train:\n', y_train)
clf = LogisticRegression(solver='lbfgs')

clf.fit(X_train, y_train)

print(f'Accuracy: {clf.score(X_train, y_train)}')

print(f'AUC ROC: {roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])}')
X_train, y_train = create_dataset(append_counts=True, decimals_ones=4, decimals_zeros=4)

print('X_train:\n', X_train)

print('y_train:\n', y_train)
clf = LogisticRegression(solver='lbfgs')

clf.fit(X_train, y_train)

print(f'Accuracy: {clf.score(X_train, y_train)}')

print(f'AUC ROC: {roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])}')
X_train, y_train = create_dataset(append_counts=True, decimals_ones=4, decimals_zeros=3)

print('X_train:\n', X_train)

print('y_train:\n', y_train)
clf = LogisticRegression(solver='lbfgs')

clf.fit(X_train, y_train)

print(f'Accuracy: {clf.score(X_train, y_train)}')

print(f'AUC ROC: {roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])}')
decimals_zeros = 3.999

X_train, y_train = create_dataset(append_counts=True, decimals_ones=4, decimals_zeros=decimals_zeros)

print('X_train:\n', X_train)

print('y_train:\n', y_train)

print(f'Equivalent as multipling by: {np.power(10, decimals_zeros)} (more optimistic case than Santander)')
clf = LogisticRegression(solver='lbfgs')

clf.fit(X_train, y_train)

print(f'Accuracy: {clf.score(X_train, y_train)}')

print(f'AUC ROC: {roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])}')
X_train, y_train = create_dataset(append_counts=False, decimals_ones=4, decimals_zeros=3)

print('X_train:\n', X_train)

print('y_train:\n', y_train)
clf = LogisticRegression(solver='lbfgs')

clf.fit(X_train, y_train)

print(f'Accuracy: {clf.score(X_train, y_train)}')

print(f'AUC ROC: {roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])}')