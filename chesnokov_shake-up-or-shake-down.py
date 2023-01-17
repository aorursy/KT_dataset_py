import numpy as np 

import pandas as pd 

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')



# load the data

oof_preds = pd.read_csv('../input/xgb-valid-preds-public/xgb_valid.csv')

y = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv', 

                usecols = ['id', 'target'])



print('Shape of OOF preds: \t', oof_preds.shape)

print('Shape of train target:\t', y.shape)
# gini calculation from https://www.kaggle.com/tezdhar/faster-gini-calculation

def ginic(actual, pred):

    actual = np.asarray(actual) #In case, someone passes Series or list

    n = len(actual)

    a_s = actual[np.argsort(pred)]

    a_c = a_s.cumsum()

    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0

    return giniSum / n

 

def gini_normalizedc(a, p):

    if p.ndim == 2:#Required for sklearn wrapper

        p = p[:,1] #If proba array contains proba for both 0 and 1 classes, just pick class 1

    return ginic(a, p) / ginic(a, a)
PROPORTION_PRIVATE = 0.70

y_preds_public, y_preds_private, y_public, y_private = train_test_split(oof_preds.target.values, 

                                                                        y.target.values, 

                                                                        test_size=PROPORTION_PRIVATE, 

                                                                        random_state=42)



print('Proportion of private:\t',PROPORTION_PRIVATE)

print('Public score:\t', round(gini_normalizedc(y_public, y_preds_public), 6))

print('Private score:\t', round(gini_normalizedc(y_private, y_preds_private), 6))
%%time

gini_public = []

gini_private = []

# do the split 10k times

for rs in range(10000):

    y_preds_public, y_preds_private, y_public, y_private = train_test_split(oof_preds.target.values, 

                                                                            y.target.values, 

                                                                            test_size=PROPORTION_PRIVATE, 

                                                                            random_state=rs)

    gini_public.append(gini_normalizedc(y_public, y_preds_public))

    gini_private.append(gini_normalizedc(y_private, y_preds_private))



# save results to numpy arrays

gini_public_arr = np.array(gini_public)

gini_private_arr = np.array(gini_private)
# 10000 random_states

plt.figure(figsize=(10,6))

plt.hist(gini_public_arr - gini_private_arr, bins=50)

plt.title('(Public - Private) scores')

plt.xlabel('Gini score difference')

plt.show()
#find indexies where public score was .284

my_indexies = np.where((gini_public_arr >= 0.284) &(gini_public_arr < 0.285))[0]



plt.figure(figsize=(10,6))

plt.hist(gini_public_arr[my_indexies] - gini_private_arr[my_indexies], bins=50)

plt.title('(Public - Private) scores, where public = .284')

plt.xlabel('Gini score difference')

plt.show()
#find indexies where public score was .286

my_indexies = np.where((gini_public_arr >= 0.286) &(gini_public_arr < 0.287))[0]



plt.figure(figsize=(10,6))

plt.hist(gini_public_arr[my_indexies] - gini_private_arr[my_indexies], bins=50)

plt.title('(Public - Private) scores, where public = .286')

plt.xlabel('Gini score difference')

plt.show()
#find indexies where public score was .284-.287

my_indexies = np.where((gini_public_arr >= 0.284) &(gini_public_arr < 0.287))[0]



plt.figure(figsize=(10,6))

plt.hist(gini_public_arr[my_indexies] - gini_private_arr[my_indexies], bins=50)

plt.title('(Public - Private) scores, where public between .284 and .287')

plt.xlabel('Gini score difference')

plt.show()