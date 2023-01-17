import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time



#This keeps the "middle" columns from being omitted when wide dataframes are being displayed

pd.options.display.max_columns = None



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



potential_features = ['ps_ind_06_bin',

                      'ps_ind_07_bin',

                      'ps_ind_08_bin',

                      'ps_ind_16_bin',

                      'ps_ind_17_bin',

                      'ps_car_08_cat',

                      'ps_ind_04_cat',

                      'ps_car_03_cat',

                      'ps_car_11_cat',

                      'ps_car_09_cat',

                      'ps_car_06_cat',

                      'ps_ind_05_cat',

                      'ps_car_05_cat',

                      'ps_car_04_cat',

                      'ps_car_01_cat',

                      'ps_car_02_cat',

                      'ps_ind_02_cat',

                      'ps_car_07_cat',

                      'ps_car_13',

                      'ps_car_12',

                      'ps_reg_02',

                      'ps_reg_03',

                      'ps_car_15',

                      'ps_reg_01',

                      'ps_ind_15',

                      'ps_ind_01',

                      'ps_car_14',

                      'ps_ind_03',

                      'ps_ind_14']



train_df = train_df[['target'] + potential_features]



#save test id's for later

test_ids = test_df['id']

test_df = test_df[potential_features]
# Compute gini

# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation

from numba import jit

@jit

def eval_gini(y_true, y_prob):

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    ntrue = 0

    gini = 0

    delta = 0

    n = len(y_true)

    for i in range(n-1, -1, -1):

        y_i = y_true[i]

        ntrue += y_i

        gini += y_i * delta

        delta += 1 - y_i

    gini = 1 - 2 * gini / (ntrue * (n - ntrue))

    return gini
#Balancing: Oversample positive class...add randomly sampled positive class by a factor of over_factor

over_factor = 0.6

row_count = round(len(train_df[train_df['target'] == 1]) * over_factor)

new_rows = train_df[train_df['target'] == 1].sample(row_count, replace=True, random_state=1)



train_df = train_df.append(new_rows, ignore_index=True)

train_df['target'].value_counts(1)
#Find columns with missing data

train_miss = []

test_miss = []

print("train columns with missing data: ")

for col in train_df.columns:

    if (train_df[col].min() == -1) and ('cat' not in col):

        train_miss.append(col)

        print (col)



print("\ntest columns with missing data: ")

for col in test_df.columns:

    if (test_df[col].min() == -1) and ('cat' not in col):

        test_miss.append(col)

        print (col)
for col in train_miss:

    print ("Train: ", col, ": ", len(train_df[train_df[col] == -1])) #count number of missing data points

    

for col in test_miss:

    print ("Test: ", col, ": ", len(test_df[test_df[col] == -1])) #count number of missing data points
#Missing Value Handling



#For categorical features, I plan to treat missing values as another category, so nothing needs to be done

#The other features with remaining missing values are 'ps_car_12', 'ps_reg_03', 'ps_car_14'

#For ps_car_12, I'll simply replace the missing values with the mean of the values present



train_df.loc[train_df['ps_car_12'] == -1, 'ps_car_12'] = train_df[train_df['ps_car_12'] != -1]['ps_car_12'].mean()
#For the other two missing features, I'll use linear regression using the most correlated features

from sklearn import linear_model

feature_mod = linear_model.LinearRegression()



from xgboost import XGBRegressor

feature_xgb = XGBRegressor(n_estimators=200)



# Fill missing features for ps_car_14, using ps_car_12 and ps_car 13

corr_list = ['ps_car_12', 'ps_car_13']

feature_mod.fit(train_df[train_df['ps_car_14'] != -1][corr_list], train_df[train_df['ps_car_14'] != -1]['ps_car_14'])

train_df.loc[train_df['ps_car_14'] == -1, 'ps_car_14'] = feature_mod.predict(train_df[train_df['ps_car_14'] == -1][corr_list])

test_df.loc[test_df['ps_car_14'] == -1, 'ps_car_14'] = feature_mod.predict(test_df[test_df['ps_car_14'] == -1][corr_list])



# Fill missing features for ps_reg_03, using ps_reg_02, ps_car_12, ps_car 13, and ps_ind_01

# Should consider trying other models to impute ps_reg_03 if that feature proves to be important

corr_list = ['ps_reg_02', 'ps_car_13', 'ps_car_12', 'ps_ind_01']

feature_xgb.fit(train_df[train_df['ps_reg_03'] != -1][corr_list], train_df[train_df['ps_reg_03'] != -1]['ps_reg_03'])

train_df.loc[train_df['ps_reg_03'] == -1, 'ps_reg_03'] = feature_xgb.predict(train_df[train_df['ps_reg_03'] == -1][corr_list])

test_df.loc[test_df['ps_reg_03'] == -1, 'ps_reg_03'] = feature_xgb.predict(test_df[test_df['ps_reg_03'] == -1][corr_list])
#Encoding

#I'll use one-hot encoding for all categorical variables except ps_car_11_cat (cardinality is too high)

cat_cols = [x for x in potential_features if '_cat' in x ]

cat_cols = list(set(cat_cols) - set(['ps_car_11_cat']))



train_df = pd.get_dummies(data=train_df, columns=cat_cols, drop_first=True)

test_df = pd.get_dummies(data=test_df, columns=cat_cols, drop_first=True)
#Next, I'll use binary encoding for ps_car_11_cat

#The following creates binary format with 0 padding for number of columns required

columns_needed = max(train_df['ps_car_11_cat'].max().item().bit_length(), 

                    test_df['ps_car_11_cat'].max().item().bit_length())

format_string = '0>'+ str(columns_needed) + 'b'



#The rest of this cool trick was inspired from here: https://stackoverflow.com/questions/46775546

#First do train_df

bin_cols = train_df['ps_car_11_cat'].apply(lambda x: format(x, format_string)).str.extractall('(\d)').unstack().astype(np.int8).add_prefix('ps_car_11_cat_b')



bin_cols.columns = bin_cols.columns.droplevel()

train_df = pd.concat([train_df, bin_cols], axis=1)

train_df.drop('ps_car_11_cat', inplace=True, axis=1)



#Now do test_df

bin_cols = test_df['ps_car_11_cat'].apply(lambda x: format(x, format_string)).str.extractall('(\d)').unstack().astype(np.int8).add_prefix('ps_car_11_cat_b')



bin_cols.columns = bin_cols.columns.droplevel()

test_df = pd.concat([test_df, bin_cols], axis=1)

test_df.drop('ps_car_11_cat', inplace=True, axis=1)
# Set up classifier

from xgboost import XGBClassifier



#MAX_ROUNDS = 400 #original

MAX_ROUNDS = 400

#LEARNING_RATE = 0.07 #original

LEARNING_RATE = 0.07



model = XGBClassifier(    

                        n_estimators=MAX_ROUNDS,

                        max_depth=4,

                        objective="binary:logistic",

                        learning_rate=LEARNING_RATE, 

                        subsample=.8,

                        min_child_weight=6,

                        colsample_bytree=.8,

                        gamma=10,

                        reg_alpha=8,

                        reg_lambda=1.3

                     )
# based on https://www.kaggle.com/aharless/xgboost-cv-lb-284 with minor modifications



from sklearn.model_selection import KFold



K = 5

kf = KFold(n_splits = K, random_state = 1, shuffle = True)



y_valid_preds = 0 * train_df['target']

y_test_preds = 0



for i, (train_index, valid_index) in enumerate(kf.split(train_df)):

    print("\nStarting fold {}".format(i+1))

    start = time.time()

    

    y_train = train_df['target'].loc[train_index]

    X_train = train_df.drop('target', axis=1).loc[train_index]

    y_valid = train_df['target'].loc[valid_index]

    X_valid = train_df.drop('target', axis=1).loc[valid_index]

      

    fit_model = model.fit(X_train, y_train, verbose=True)

    

    preds = fit_model.predict_proba(X_valid)[:, 1]

    gini = eval_gini(y_valid, preds)

    y_valid_preds.loc[valid_index] = preds



    y_test_preds += fit_model.predict_proba(test_df)[:, 1]

    

    print("Fold {} Gini score: {}".format(i, gini))

    print("Completed fold {} in {:.2f} minutes\n".format(i+1, (time.time() - start)/60))



y_test_preds /= K



print( "\nGini for full training set:" )

eval_gini(train_df['target'], y_valid_preds)
from xgboost import plot_importance

import matplotlib.pyplot as plt

%matplotlib inline



# plot feature importance

fig, ax = plt.subplots(figsize=(14, 18))

plot_importance(fit_model, ax=ax)
# Create submission file

sub = pd.DataFrame()

sub['id'] = test_ids

sub['target'] = y_test_preds

sub.to_csv('xgb_submit_3.csv', float_format='%.6f', index=False)