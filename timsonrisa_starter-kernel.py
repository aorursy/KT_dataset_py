import os

from tqdm import tqdm

import itertools



import numpy as np

import pandas as pd



from collections import OrderedDict



from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GroupKFold

from sklearn.metrics import mean_absolute_error
# Import the training data and inspect the first few entries:

train_data = pd.read_csv("../input/train.csv", index_col='id')

y_label = train_data.pop('scalar_coupling_constant')

print(f"Training data is of shape: {train_data.shape}")

train_data.head(3)
# Import the test data and inspect the first few entries:

test_data = pd.read_csv("../input/test.csv", index_col='id')

print(f"Test data is of shape: {test_data.shape}")

test_data.head(3)
# Quick Inspections on the training and test data:

print(f"Training Data has {train_data.molecule_name.nunique()} unique molecules with {train_data.type.nunique()} unique types")

print(f"Test Data has {test_data.molecule_name.nunique()} unique molecules with {test_data.type.nunique()} unique types")

print(f"Coupling Constant Dist.: mean={round(y_label.mean(),2)} ± std={round(y_label.std(),2)}")
# Inspection on the structures data set:

structures = pd.read_csv("../input/structures.csv")

structures.head(3)
# Merge the coordinates of both atoms to the training and test data:

def MergeData(data, structures):

    

    for i in range(2):

        data = pd.merge(data, structures, how = "inner", left_on = ["molecule_name", f"atom_index_{i}"], right_on = ["molecule_name", "atom_index"])

        data.drop(columns=["atom_index"], inplace=True)

        data.rename(index=str, columns={"atom": f"atom_{i}", "x": f"x_{i}", "y": f"y_{i}", "z": f"z_{i}"}, inplace=True)

    

    data = data.reindex(columns=['molecule_name', 'type', 'atom_index_0', 'atom_0', 'x_0', 'y_0', 'z_0', 'atom_index_1', 'atom_1', 'x_1', 'y_1', 'z_1'])

    return data



train_data = MergeData(train_data, structures)

test_data = MergeData(test_data, structures)
# Encode the categorical variables: molecule type and atoms names:

for f in ['type', 'atom_0', 'atom_1']:

    lbl = LabelEncoder()

    lbl.fit(list(train_data[f].values) + list(test_data[f].values))

    train_data[f] = lbl.transform(list(train_data[f].values))

    test_data[f] = lbl.transform(list(test_data[f].values))
# Calculate the distance between two atoms:

train_p_0 = train_data[['x_0', 'y_0', 'z_0']].values

train_p_1 = train_data[['x_1', 'y_1', 'z_1']].values

test_p_0 = test_data[['x_0', 'y_0', 'z_0']].values

test_p_1 = test_data[['x_1', 'y_1', 'z_1']].values



train_data['distance'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test_data['distance'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
molecules = train_data.pop('molecule_name')

test_data = test_data.drop('molecule_name', axis=1)
# Simple Regressor, using only the encoded values:

feature_columns = ['type', 'atom_0', 'atom_index_0', 'x_0', 'y_0', 'z_0',

                   'atom_index_1', 'atom_1', 'x_1', 'y_1', 'z_1', 'distance']

train = train_data[feature_columns]

test = test_data[feature_columns]
def metric(df, prediction, labels):

    df = df.copy()

    df.loc[:, "prediction"] = prediction

    df.loc[:, "scalar_coupling_constant"] = labels

    maes = []

    for t in df.type.unique():

        y_true = df[df.type==t].scalar_coupling_constant.values

        y_pred = df[df.type==t].prediction.values        

        mae = np.log(mean_absolute_error(y_true, y_pred))

        # print(f"MAE for type {t} is {round(mae, 2)}")

        maes.append(mae)

    return np.mean(maes)
'''

yoof = np.zeros(len(train))

yhat = np.zeros(len(test))



n_splits = 3

gkf = GroupKFold(n_splits=n_splits)



fold = 0

for in_index, oof_index in gkf.split(train, y_label, groups=molecules):

    fold += 1

    print(f'fold {fold} of {n_splits}')

    X_in, X_oof = train.values[in_index], train.values[oof_index]

    y_in, y_oof = y_label.values[in_index], y_label.values[oof_index]

    

    reg = RandomForestRegressor(n_estimators=1,

                                max_depth=5,

                                min_samples_leaf=3,

                                n_jobs=-1)

    reg.fit(X_in, y_in)

    yoof[oof_index] = reg.predict(X_oof)

    

    cur_fold_mea = metric(train[oof_index], yoof[oof_index], y_label[oof_index].values)

    print(f"The MEA for current fold is {round(cur_fold_mea, 2)}")



    cur_yhat = reg.predict(test)

    yhat += cur_yhat



yhat /= n_splits

'''
# Create the Hyper Parameters Grid Search:

max_features = ['sqrt']  #log2

max_depth = [12]

properties_list = list(itertools.product(max_features, max_depth))
# Create Regressor for every parameters combination:

regressors = []

for cur_prop in properties_list:

    regressors.append(("RF: {} max feat + {} max depth".format(*cur_prop),

                        RandomForestRegressor(warm_start=True,

                                               max_features=cur_prop[0],

                                               max_depth=cur_prop[1])))



# Map a regressor name to a list of (<n_estimators>, <error rate>) pairs.

oof_error_rate = OrderedDict((label, []) for label, _ in regressors)
# Range of `n_estimators` values to explore.

min_estimators = 1

max_estimators = 101

step_estimators = 50



gkf = GroupKFold(n_splits=2)

in_index, oof_index = gkf.split(train, y_label, groups=molecules)

in_index = in_index[0]

oof_index = oof_index[0]

        

for label, reg in tqdm(regressors, desc="Regressors Loop"):

    for i in tqdm(range(min_estimators, max_estimators + 1, step_estimators), desc="Trees Loop"):

        

        reg.set_params(n_estimators=i)

        reg.fit(train.values[in_index], y_label.values[in_index])

        yoof = reg.predict(train.values[oof_index])



        # Record the Validation Error for each `n_estimators=i` setting.

        oof_error = metric(train.iloc[oof_index], yoof, y_label.values[oof_index])

        print(f"Current OOF Error is {oof_error}")

        oof_error_rate[label].append((i, oof_error))
print(oof_error_rate)
# print(f"Final Metric for all Out Of Sample data: {round(metric(train, yoof, y_label.values),2)}")
'''

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='id')

benchmark = sample_submission.copy()

benchmark['scalar_coupling_constant'] = yhat

benchmark.to_csv('simple_benchmark.csv')

'''