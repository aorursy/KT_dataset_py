# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input/prepare-data-for-decision-trees-algorithms"))

# Any results you write to the current directory are saved as output.
# Now load the training set
print('Loading the training set:')
dtypes = {
    'MachineIdentifier': 'category',
    'AVProductsInstalled': 'float32',
    'CountryIdentifier': 'float32',
    'OrganizationIdentifier': 'float32',
    'GeoNameIdentifier': 'float32',
    'LocaleEnglishNameIdentifier': 'float32',
    'OsBuild': 'int16',
    'OsSuite': 'float32',
    'OsPlatformSubRelease': 'float32',
    'SkuEdition': 'float32',
    'IeVerIdentifier': 'float32',
    'SmartScreen': 'float32',
    'Census_MDC2FormFactor': 'float32',
    'Census_ProcessorCoreCount': 'float16',
    'Census_ProcessorManufacturerIdentifier': 'float32',
    'Census_PrimaryDiskTotalCapacity': 'float32',
    'Census_PrimaryDiskTypeName': 'float32',
    'Census_SystemVolumeTotalCapacity': 'float32',
    'Census_TotalPhysicalRAM': 'float32',
    'Census_ChassisTypeName': 'float32',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches': 'float16',
    'Census_InternalPrimaryDisplayResolutionHorizontal': 'float16',
    'Census_InternalPrimaryDisplayResolutionVertical': 'float16',
    'Census_PowerPlatformRoleName': 'float32',
    'Census_InternalBatteryType': 'float32',
    'Census_InternalBatteryNumberOfCharges': 'float32',
    'Census_OSBranch': 'float32',
    'Census_OSBuildNumber': 'int16',
    'Census_OSBuildRevision': 'int32',
    'Census_OSEdition': 'float32',
    'Census_OSSkuName': 'float32',
    'Census_OSInstallTypeName': 'float32',
    'Census_OSInstallLanguageIdentifier': 'float32',
    'Census_OSUILocaleIdentifier': 'float32',
    'Census_OSWUAutoUpdateOptionsName': 'float32',
    'Census_GenuineStateName': 'float32',
    'Census_ActivationChannel': 'float32',
    'Census_IsFlightingInternal': 'float32',
    'Census_ThresholdOptIn': 'float16',
    'Census_IsSecureBootEnabled': 'int8',
    'Census_IsWIMBootEnabled': 'float32',
    'Census_IsTouchEnabled': 'int8',
    'Wdft_IsGamer': 'float32',
    'Wdft_RegionIdentifier': 'float32',
    'HasDetections': 'int8',
    'EngineVersion_0': 'float32',
    'EngineVersion_1': 'float32',
    'EngineVersion_2': 'float32',
    'EngineVersion_3': 'float32',
    'AppVersion_0': 'float32',
    'AppVersion_1': 'float32',
    'AppVersion_2': 'float32',
    'AppVersion_3': 'float32',
    'AvSigVersion_0': 'float32',
    'AvSigVersion_1': 'float32',
    'AvSigVersion_2': 'float32',
    'AvSigVersion_3': 'float32',
    'Census_OSVersion_0': 'float32',
    'Census_OSVersion_1': 'float32',
    'Census_OSVersion_2': 'float32',
    'Census_OSVersion_3': 'float32'
}
training_set = pd.read_csv('../input/prepare-data-for-decision-trees-algorithms/training_decisionTrees.csv', dtype=dtypes)
print('Training set loaded')
print(training_set.shape)

# Handle Nans
def create_nan_dict(data):
    ret = dict()
    for col in data:
        if col != 'HasDetections' and col != 'MachineIdentifier':
            ret[col] = data[col].astype('float32').mean()
    return ret


def remove_nans(data, nan_dict):
    for col in data:
        if col != 'HasDetections' and col != 'MachineIdentifier':
            data[col] = data[col].fillna(nan_dict[col])

print('Handling NaN in training set')
nan_dict = create_nan_dict(training_set)
remove_nans(training_set, nan_dict)
print('Done')
columns_to_remove = []
for col_name in training_set.columns.values:
    if col_name == 'HasDetections' or col_name == 'MachineIdentifier':
        continue
    unique_values = training_set[col_name].value_counts(dropna=False)
    msg = 'column ' + col_name + ' have ' + str(len(unique_values)) + ' unique values. The bigger category has ' + str(100 * unique_values.values[0] / training_set.shape[0]) + ' percent of the data'
    if len(unique_values)==1:
        msg = msg + " - removed"
        del training_set[col_name]  
        columns_to_remove.append(col_name)
    print(msg)

print('')
print('Untill now ' + str(len(columns_to_remove)) + ' colums removed')
print(training_set.shape)
def fine_tune_decision_tree(training_set, k_fold):
    results = dict()
    avg_grade = dict()
    std_grade = dict()
    min_grade = dict()
    max_grade = dict()
    best_sample_leaf = 0
    best_grade = 0.5
    for min_samples_leaf in [200, 400, 500, 600, 800, 1000]:
        features = [c for c in training_set.columns if c not in ['MachineIdentifier', 'HasDetections']]
        dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
        results[min_samples_leaf] = []

        # Train and tests the data on k_fold splits and store the results
        for train_indices, test_indices in k_fold.split(training_set):
            print('Start fitting')
            dt.fit(training_set[features].iloc[train_indices], training_set['HasDetections'].iloc[train_indices])
            print('End fitting - Start predicting')
            prob = dt.predict_proba(training_set[features].iloc[test_indices])
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(training_set['HasDetections'].iloc[test_indices], prob[:, 1])
            results[min_samples_leaf].append(sklearn.metrics.auc(fpr, tpr))
            print('End predicting')

        grade = np.mean(results[min_samples_leaf])
        if grade > best_grade:
            best_grade = grade
            best_sample_leaf = min_samples_leaf
        avg_grade[min_samples_leaf] = grade
        std_grade[min_samples_leaf] = np.std(results[min_samples_leaf])
        min_grade[min_samples_leaf] = np.min(results[min_samples_leaf])
        max_grade[min_samples_leaf] = np.max(results[min_samples_leaf])

    # Now plot the result.
    n_leafs = avg_grade.keys()
    avgs = [avg_grade[l] for l in n_leafs]
    stds = [std_grade[l] for l in n_leafs]
    plt.figure()
    plt.errorbar(n_leafs, avgs, stds)
    plt.title('decision tree classifier k fold results')
    plt.xlabel('number of minimum sample in a leaf')
    plt.ylabel('ROC curve area')
    plt.show()

    return DecisionTreeClassifier(min_samples_leaf=best_sample_leaf)

# Define decision tree predictor and fine tune its variables
k_fold = KFold(n_splits=5, shuffle=True)
classifier = fine_tune_decision_tree(training_set, k_fold)
features = [c for c in training_set.columns if c not in ['MachineIdentifier', 'HasDetections']]
X = training_set[features]
y = training_set['HasDetections']
del training_set
classifier.fit(X, y)
del X
del y
test = pd.read_csv('../input/prepare-data-for-decision-trees-algorithms/test_decisionTrees.csv', dtype=dtypes)
remove_nans(test, nan_dict)

X = test[features]
y_pred = classifier.predict_proba(X)[:, 1]
to_submit = pd.DataFrame(test['MachineIdentifier'])
to_submit['HasDetections'] = y_pred
to_submit.to_csv('decisionTreeClassifierRes.csv', index=False)