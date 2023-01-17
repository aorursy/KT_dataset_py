%matplotlib inline



import pandas as pd

from sklearn.ensemble import RandomForestClassifier, forest

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

from IPython.display import display

import numpy as np

import scipy

import re



# Permutation Importance

import eli5

from eli5.sklearn import PermutationImportance



# Partial Plots

from pdpbox import pdp, get_dataset, info_plots



# Package used to calculate SHAP Values

import shap
dtypes = {

        'MachineIdentifier':                                    'category',

        'ProductName':                                          'category',

        'EngineVersion':                                        'category',

        'AppVersion':                                           'category',

        'AvSigVersion':                                         'category',

        'IsBeta':                                               'int8',

        'RtpStateBitfield':                                     'float16',

        'IsSxsPassiveMode':                                     'int8',

        'DefaultBrowsersIdentifier':                            'float16',

        'AVProductStatesIdentifier':                            'float32',

        'AVProductsInstalled':                                  'float16',

        'AVProductsEnabled':                                    'float16',

        'HasTpm':                                               'int8',

        'CountryIdentifier':                                    'int16',

        'CityIdentifier':                                       'float32',

        'OrganizationIdentifier':                               'float16',

        'GeoNameIdentifier':                                    'float16',

        'LocaleEnglishNameIdentifier':                          'int8',

        'Platform':                                             'category',

        'Processor':                                            'category',

        'OsVer':                                                'category',

        'OsBuild':                                              'int16',

        'OsSuite':                                              'int16',

        'OsPlatformSubRelease':                                 'category',

        'OsBuildLab':                                           'category',

        'SkuEdition':                                           'category',

        'IsProtected':                                          'float16',

        'AutoSampleOptIn':                                      'int8',

        'PuaMode':                                              'category',

        'SMode':                                                'float16',

        'IeVerIdentifier':                                      'float16',

        'SmartScreen':                                          'category',

        'Firewall':                                             'float16',

        'UacLuaenable':                                         'float32',

        'Census_MDC2FormFactor':                                'category',

        'Census_DeviceFamily':                                  'category',

        'Census_OEMNameIdentifier':                             'float16',

        'Census_OEMModelIdentifier':                            'float32',

        'Census_ProcessorCoreCount':                            'float16',

        'Census_ProcessorManufacturerIdentifier':               'float16',

        'Census_ProcessorModelIdentifier':                      'float16',

        'Census_ProcessorClass':                                'category',

        'Census_PrimaryDiskTotalCapacity':                      'float32',

        'Census_PrimaryDiskTypeName':                           'category',

        'Census_SystemVolumeTotalCapacity':                     'float32',

        'Census_HasOpticalDiskDrive':                           'int8',

        'Census_TotalPhysicalRAM':                              'float32',

        'Census_ChassisTypeName':                               'category',

        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',

        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',

        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',

        'Census_PowerPlatformRoleName':                         'category',

        'Census_InternalBatteryType':                           'category',

        'Census_InternalBatteryNumberOfCharges':                'float32',

        'Census_OSVersion':                                     'category',

        'Census_OSArchitecture':                                'category',

        'Census_OSBranch':                                      'category',

        'Census_OSBuildNumber':                                 'int16',

        'Census_OSBuildRevision':                               'int32',

        'Census_OSEdition':                                     'category',

        'Census_OSSkuName':                                     'category',

        'Census_OSInstallTypeName':                             'category',

        'Census_OSInstallLanguageIdentifier':                   'float16',

        'Census_OSUILocaleIdentifier':                          'int16',

        'Census_OSWUAutoUpdateOptionsName':                     'category',

        'Census_IsPortableOperatingSystem':                     'int8',

        'Census_GenuineStateName':                              'category',

        'Census_ActivationChannel':                             'category',

        'Census_IsFlightingInternal':                           'float16',

        'Census_IsFlightsDisabled':                             'float16',

        'Census_FlightRing':                                    'category',

        'Census_ThresholdOptIn':                                'float16',

        'Census_FirmwareManufacturerIdentifier':                'float16',

        'Census_FirmwareVersionIdentifier':                     'float32',

        'Census_IsSecureBootEnabled':                           'int8',

        'Census_IsWIMBootEnabled':                              'float16',

        'Census_IsVirtualDevice':                               'float16',

        'Census_IsTouchEnabled':                                'int8',

        'Census_IsPenCapable':                                  'int8',

        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',

        'Wdft_IsGamer':                                         'float16',

        'Wdft_RegionIdentifier':                                'float16',

        'HasDetections':                                        'int8'

        }



%time train = pd.read_csv('../input/train.csv', nrows=1000000, usecols=dtypes.keys(), low_memory=False)



#display(train.describe(include='all').T)
col = ['EngineVersion', 'AppVersion', 'AvSigVersion', 'OsBuildLab', 'Census_OSVersion']

for c in col:

    for i in range(6):

        train[c + str(i)] = train[c].map(lambda x: re.split('\.|-', str(x))[i] if len(re.split('\.|-', str(x))) > i else -1)

        try:

            train[c + str(i)] = pd.to_numeric(train[c + str(i)])

        except:

            #print(f'{c + str(i)} cannot be casted to number')

            pass

            

train['HasExistsNotSet'] = train['SmartScreen'] == 'ExistsNotSet'

#In the competition details, a strong time component was indicated. 

#At this point, I am not aware of any columns which show this time component, so lets for now split our validation set based on the index

def split_train_val_set(X, Y, n):

    if n < 1: n=int(len(X.index) * n)

    return X.iloc[:n], X.iloc[n:], Y.iloc[:n], Y.iloc[n:]



#We prepare the training data by replacing the category variables with the category codes 

#and replacing the nan values in the numerical columns with the median

for col, val in train.items():

    if pd.api.types.is_string_dtype(val): 

        train[col] = val.astype('category').cat.as_ordered()

        train[col] = train[col].cat.codes

    elif pd.api.types.is_numeric_dtype(val) and val.isnull().sum() > 0:

        train[col] = val.fillna(val.median())



X, Y = train.drop('HasDetections', axis=1), train['HasDetections']

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

#X_train, X_val, Y_train, Y_val = split_train_val_set(X, Y, n=0.1)

X_train.head(5)



#To be able to test the models rapidly, we create a function to print the scores of the model.

def print_score(m):

    res = [roc_auc_score(m.predict(X_train), Y_train), roc_auc_score(m.predict(X_val), Y_val), 

           m.score(X_train, Y_train), m.score(X_val, Y_val)

          ]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)

    

#As in the fastai course, the rf_samples can be reduced to allow for faster repetition cycles. 

#We also immediately create a reset function to check the model performance on the entire dataset.

def set_rf_samples(n):

    """ Changes Scikit learn's random forests to give each tree a random sample of

    n random rows.

    """

    forest._generate_sample_indices = (lambda rs, n_samples: forest.check_random_state(rs).randint(0, n_samples, n))

    

def reset_rf_samples():

    """ Undoes the changes produced by set_rf_samples.

    """

    forest._generate_sample_indices = (lambda rs, n_samples:

        forest.check_random_state(rs).randint(0, n_samples, n_samples))

    

set_rf_samples(50000)

train[:5]
train.describe()
model = RandomForestClassifier(n_estimators=100, min_samples_leaf=50, max_features=0.5, n_jobs=-1, oob_score=False)

%time model.fit(X_train, Y_train)



print_score(model)
perm = PermutationImportance(model, random_state=1).fit(X_val, Y_val)

eli5.show_weights(perm, feature_names = X_val.columns.tolist())
feat_names = ['AVProductStatesIdentifier', 'AVProductsInstalled', 'AvSigVersion']



for feat_name in feat_names:

    pdp_dist = pdp.pdp_isolate(model=model, dataset=X_val, model_features=X_val.columns.tolist(), feature=feat_name)

    pdp.pdp_plot(pdp_dist, feat_name)

    plt.show()
inter1  =  pdp.pdp_interact(model=model, dataset=X_val, model_features=X_val.columns.tolist(), features=['AVProductStatesIdentifier', 'AVProductsInstalled'])



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['AVProductStatesIdentifier', 'AVProductsInstalled'], plot_type='contour')

plt.show()
row_to_show = 17

data_for_prediction = X_val.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)



model.predict_proba(data_for_prediction_array)
# Create object that can calculate shap values

explainer = shap.TreeExplainer(model)



# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)



shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.

shap_values = explainer.shap_values(X_val)



# Make plot. Index of [1] is explained in text below.

shap.summary_plot(shap_values[1], X_val)