import pandas as pd
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
from scipy import stats
train = pd.read_csv('../input/application_train.csv')
final = pd.read_csv('../input/application_test.csv')
train_1 = train.iloc[:,1:] #discard theSKID_CURR first
final_1 = final.iloc[:,1:]
cat_cols = [f for f in train_1.columns if train_1[f].dtype == 'object']
train_2 = pd.get_dummies(train_1, columns=cat_cols)
final_2 = pd.get_dummies(final_1, columns=cat_cols)

target = np.array(train_2.iloc[:,0])
train_2 = train_2.iloc[:,1:]
def check_dtype(df):
    for i in range(df.shape[1]):
        dtype_list = ['int64', 'float64', 'uint8']
        datatype = df.iloc[:,i].dtype
        if not(datatype in dtype_list):
            print('datatype')
#no print -> only these three dtype

check_dtype(train_2)
check_dtype(final_2)
def neg_search(train_2):
    neg_list = []
    neg_pos_list = []
    neg_missing_list = []
    neg_only_list = []

    for i in range(train_2.shape[1]):
        if np.sum(train_2.iloc[:,i] < 0) > 0:
            neg_list.append(train_2.columns[i])

    for item in neg_list:
        if (np.sum(np.isnan(train_2[item])) > 0):
            #print(item +  ' includes missing')
            neg_missing_list.append(item)

    for item in neg_list:
        if (np.sum(train_2[item] > 0) > 0):
            #print(item +  ' includes positive value')
            neg_pos_list.append(item)

    for item in neg_list:
        if not(item in neg_pos_list) and not(item in neg_missing_list):
            neg_only_list.append(item)
            #print(item + 'includes negative value only')
        
    return neg_pos_list, neg_missing_list, neg_only_list
def neg_transform(test, train, neg_only_list, neg_pos_list):
    missing_list = []
    for item in neg_only_list:
        test[item] = test[item] * (-1)
    for item in neg_pos_list:
        test[item] = test[item] - np.min(train[item])

    return test
def missing_transform(test, train):
    for i in range(test.shape[1]):
        if (np.sum(np.isnan(test.iloc[:,i])) > 0) & (np.sum(np.isnan(test.iloc[:,i])) < 1000):
            test.iloc[:,i][np.isnan(test.iloc[:,i])] = np.nanmedian(train.iloc[:,i])
        #elif np.sum(np.isnan(test.iloc[:,i])) > 1000:
            #print(str(train.iloc[:,i].dtype) + ', ' + test.columns[i] + ' contains ' + str(np.sum(np.isnan(test.iloc[:,i]))) + ' missing value')
    return test
def normal_transform(test, train):
    for i in range(train.shape[1]):
        if (((train.iloc[:,i].dtype == 'int64')&(np.max(train.iloc[:,i]) > 10))|(train.iloc[:,i].dtype == 'float64')):
            if np.max(train.iloc[:,i]) / 3 >= np.min(train.iloc[:,i]):
                test.iloc[:,i] = np.log(test.iloc[:,i] + 1)

    return test
def boostrapping(df, num_round = 100):
    size = int(df.shape[0] / 10)
    total_mean = np.nanmean(np.random.choice(df, size, replace=True))
    total_std = np.nanstd(np.random.choice(df, size, replace=True))
    for k in range(int(num_round - 1)):
        total_mean = total_mean + np.nanmean(np.random.choice(df, size, replace=True))
        total_std = total_std + np.nanstd(np.random.choice(df, size, replace=True))
    report_mean = total_mean / num_round
    report_std = total_std / num_round
    return report_mean, report_std
neg_pos_list, neg_missing_list, neg_only_list = neg_search(train_2)
final_2 = neg_transform(final_2, train_2, neg_only_list, neg_pos_list)
train_2 = neg_transform(train_2, train_2, neg_only_list, neg_pos_list)

final_2 = missing_transform(final_2, train_2)
train_2 = missing_transform(train_2, train_2)

final_2 = normal_transform(final_2, train_2)
train_2 = normal_transform(train_2, train_2)
permutation = np.random.permutation(train_2.shape[0])
test_index = permutation[:int(len(permutation) * 0.001)]
train_index = permutation[int(len(permutation) * 0.001):]

test_label = target[test_index]
test = train_2.iloc[test_index, 1:]

train_y1 = train_2.iloc[train_index, :][target[train_index] == 1]
train_y0 = train_2.iloc[train_index, :][target[train_index] == 0]
dist = {}
for i in range(train_y1.shape[1]):
    if (train_y1.iloc[:,i].dtype == 'int64'):
        if np.max(train_y1.iloc[:,i]) < 10:
            dist[train_y1.columns[i]] = 'poisson'
        else:
            dist[train_y1.columns[i]] = 'normal'
    elif train_y1.iloc[:,i].dtype == 'float64':
        dist[train_y1.columns[i]] = 'normal'
    elif train_y1.iloc[:,i].dtype == 'uint8':
        dist[train_y1.columns[i]] = 'binomial'
start_time = time.time()

posterior_y1 = {}
posterior_y0 = {}
for i in range(train_y1.shape[1]):
    if np.sum(train_y0.iloc[:,i]) > 0:
        if dist[train_y1.columns[i]] == 'poisson':
            boostrap_mean, boostrap_std = boostrapping(train_y1.iloc[:,i])
            posterior_y1[train_y1.columns[i]] = boostrap_mean

            boostrap_mean, boostrap_std = boostrapping(train_y0.iloc[:,i], num_round = 15)
            posterior_y0[train_y1.columns[i]] = boostrap_mean

        elif dist[train_y1.columns[i]] == 'normal':
            missing = np.sum(np.isnan(train_y1.iloc[:,i])) / train_y1.shape[0]
            boostrap_mean, boostrap_std = boostrapping(train_y1.iloc[:,i])
            if boostrap_std < boostrap_mean * 0.01:
                posterior_y1[train_y1.columns[i]] = (missing, boostrap_mean, boostrap_mean * 0.01)
            else:
                posterior_y1[train_y1.columns[i]] = (missing, boostrap_mean, boostrap_std)
                
            missing = np.sum(np.isnan(train_y0.iloc[:,i])) / train_y0.shape[0]
            boostrap_mean, boostrap_std = boostrapping(train_y0.iloc[:,i], num_round = 15)
            if boostrap_std < boostrap_mean * 0.01:
                posterior_y0[train_y1.columns[i]] = (missing, boostrap_mean, boostrap_mean * 0.01)
            else:
                posterior_y0[train_y1.columns[i]] = (missing, boostrap_mean, boostrap_std)
                
        elif dist[train_y1.columns[i]] == 'binomial':
            boostrap_mean, boostrap_std = boostrapping(train_y1.iloc[:,i])
            posterior_y1[train_y1.columns[i]] = boostrap_mean

            boostrap_mean, boostrap_std = boostrapping(train_y0.iloc[:,i], num_round = 15)
            posterior_y0[train_y1.columns[i]] = boostrap_mean
            
    else:
        dist[train_y1.columns[i]] = 'noinfo'
            
finish_time = time.time()
print('time for parameter learning with boostrapping: ' + str(finish_time - start_time) + 's')
def algorithm(test, gradient_vanishing = 1, verbo_freq = 20):
    total_start = time.time()
    sample_start = time.time()
    target = []
    P_1_library = []
    P_0_library = []
    
    for j in range(test.shape[0]):

        P_1 = train_y1.shape[0] / (train_y1.shape[0] + train_y0.shape[0]) * gradient_vanishing
        P_0 = 1 * gradient_vanishing - P_1
        
        constant_1_2pi = abs(1/np.power(2* math.pi, 0.5))

        for i in range(test.shape[1]):
            if not(np.isnan(test.iloc[j,i])):
                if dist[test.columns[i]] == 'poisson':
                    mean = posterior_y1[test.columns[i]]
                    step2 = np.power(mean, test.iloc[j,i]) * np.exp(-mean)/math.factorial(test.iloc[j,i])
                    P_1 = round(P_1 * step2, 64)

                    mean = posterior_y0[test.columns[i]]
                    step2 = np.power(mean, test.iloc[j,i]) * np.exp(-mean)/math.factorial(test.iloc[j,i])
                    P_0 = round(P_0 * step2, 64)
                
                elif dist[test.columns[i]] == 'normal':
                    missing = posterior_y0[test.columns[i]][0]
                    mean = posterior_y0[test.columns[i]][1]
                    std = posterior_y0[test.columns[i]][2]
                    step1 = stats.norm.cdf(test.iloc[j,i], mean, std)
                    if step1 > 0.5:
                        step2 = (step1 - 0.5) * 2
                    else:
                        step2 = step1 * 2
                    P_0 = round(P_0 * step2 * (1 - missing), 64)
                    
                    missing = posterior_y1[test.columns[i]][0]
                    mean = posterior_y1[test.columns[i]][1]
                    std = posterior_y1[test.columns[i]][2]
                    step1 = stats.norm.cdf(test.iloc[j,i], mean, std)
                    if step1 > 0.5:
                        step2 = (step1 - 0.5) * 2
                    else:
                        step2 = step1 * 2
                    P_1 = round(P_1 * step2 * (1 - missing), 64)
                
                elif dist[test.columns[i]] == 'binomial':
                    if (test.iloc[j,i] == 1):
                        P_1 = P_1 * posterior_y1[test.columns[i]]
                        P_0 = P_0 * posterior_y0[test.columns[i]]
                    else:
                        P_1 = P_1 * (1 - posterior_y1[test.columns[i]])
                        P_0 = P_0 * (1 - posterior_y0[test.columns[i]])
            
            else:
                if dist[test.columns[i]] == 'normal':
                    P_1 = P_1 * posterior_y1[test.columns[i]][0]
                    P_0 = P_0 * posterior_y0[test.columns[i]][0]
        P_1_library.append(P_1)
        P_0_library.append(P_0)
        if P_1 > P_0:
            target.append(1)
        else:
            if P_1 == 0:
                target.append(1)
            else:
                target.append(0)
        if int(j+1)%verbo_freq == 0:
            sample_finish = time.time()
            print(str(int(j/verbo_freq+1)) + ' mini-batch finished with '+ str(sample_finish - sample_start) + 's')
            sample_start = time.time()
    total_finish = time.time()
    print('the whole prediction process takes ' + str(total_finish - total_start) + 's')
    print('average time for prediction is ' + str((total_finish - total_start)/test.shape[0]) + 's')
    
    return target, P_1_library, P_0_library
target, P_1_library, P_0_library = algorithm(test)
np.sum(target + test_label == 2) / np.sum(target)