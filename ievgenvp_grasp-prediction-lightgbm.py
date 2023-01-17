import pandas as pd

import numpy as np

import lightgbm as lgb



from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
# data load

df = pd.read_csv("../input/shadow_robot_dataset.csv")

# columns include spaces. Following code remove them:

df.columns = df.columns.str.replace('\s+', '')

print (df.head(5))
# exclude positioning. It is a subject for further debates.

no_pos = [x for x in df.columns if x[-3:] != "pos"]

print (no_pos)
# create X and Y

x_columns = [x for x in no_pos if x not in ['experiment_number',

                                                'robustness',

                                                'measurement_number']]

X = df[x_columns].copy().values

Y = df['robustness'].copy().values

Y[Y<100] = 0

Y[Y>=100] = 1

print (stats.describe(Y))



# split into train and validation datasets

seed = 32

np.random.seed(seed)

X_tr, X_val, Y_tr, Y_val = train_test_split(X, Y, test_size=0.20, random_state=seed)
# predictions with lgbm

lgbm_cl = lgb.LGBMClassifier(num_iterations=700, num_leaves=127, 

                             min_data_in_leaf=50, is_unbalance = False,

                             application='binary', metric='binary_error')



print ("Fitting training set...")

lgbm_cl.fit(X_tr, Y_tr)

print ("Done")
print ("Predicting on validation set...")

print ("")

val_pred = lgbm_cl.predict(X_val)

print (val_pred.shape)

print ("")

print (stats.describe(val_pred))

print ("")

print (val_pred)
score = accuracy_score(Y_val, val_pred)

print ("Score: %f" %(score))
x_columns_pos = [x for x in df.columns if x not in ['experiment_number',

                                                    'robustness',

                                                    'measurement_number']]

X_pos = df[x_columns_pos].copy().values

X_tr, X_val, Y_tr, Y_val = train_test_split(X_pos, Y, test_size=0.20, random_state=seed)

print ("Fitting training set...")

lgbm_cl.fit(X_tr, Y_tr)

print ("Done")
print ("Predicting on validation set...")

print ("")

val_pred = lgbm_cl.predict(X_val)

score = accuracy_score(Y_val, val_pred)

print ("Score: %f" %(score))