TEST_TRAIN_AND_PREDICT = False

PLOT_LEARNING_CURVES = False

GENERATE_SUBMISSION = True



WITH_SINGLE_PROVINCE_COUNTRY = False

SINGLE_PROVINCE_COUNTRY = ('', 'Brazil')



CASES_REG_ALPHA = 5.0

CASES_NUM_LEAVES = 8

FATALITIES_REG_ALPHA = 5.0

FATALITIES_NUM_LEAVES = 4



SEED = 654321



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



PUBLIC_LEADERBOARD_FROM_DAY = pd.Timestamp('2020-04-02') # should be 2020-04-01

PUBLIC_LEADERBOARD_TO_DAY = pd.Timestamp('2020-04-15')

PUBLIC_LEADERBOARD_USE_REAL_DATA_UP_TO = pd.Timestamp('2020-03-31')



PRIVATE_LEADERBOARD_FROM_DAY = pd.Timestamp('2020-04-16')

PRIVATE_LEADERBOARD_TO_DAY = pd.Timestamp('2020-05-14')

PRIVATE_LEADERBOARD_USE_REAL_DATA_UP_TO = pd.Timestamp('2020-04-14')

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
import random

random.seed(SEED)
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")



health_systems = pd.read_csv('/kaggle/input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv')



region_metadata = pd.read_csv('/kaggle/input/covid19-forecasting-metadata/region_metadata.csv')



#Fix Diamond Princess cruise ship area from source 

region_metadata.loc[region_metadata.Country_Region == 'Diamond Princess', 'area'] = 0.37

region_metadata.loc[region_metadata.Country_Region == 'Holy See', 'area'] = 0.44
train_data.sample(10)
merged_data = (train_data

               .merge(region_metadata

                      .rename(columns={'population': 'Population',

                                       'area': 'Area'})

                      .drop(columns=['lat',

                                     'lon',

                                     'continent',

                                     'density']),

                      how='left')

               .merge(health_systems

                      .rename(columns={'Health_exp_per_capita_USD_2016': 'HealthExpPerCapita',

                                       'Physicians_per_1000_2009-18': 'PhysiciansPer1000',

                                       'Nurse_midwife_per_1000_2009-18': 'NursesPer1000'})

                      .drop(columns=['World_Bank_Name',

                                     'Health_exp_pct_GDP_2016',

                                     'Health_exp_public_pct_2016',

                                     'Health_exp_out_of_pocket_pct_2016',

                                     'per_capita_exp_PPP_2016',

                                     'External_health_exp_pct_2016',

                                     'Specialist_surgical_per_1000_2008-18',

                                     'Completeness_of_birth_reg_2009-18',

                                     'Completeness_of_death_reg_2008-16']),

                      how='left'))
merged_data.sample(10)
def create_cases_per_population(dataset):

    dataset.loc[:, 'CasesPerPopulation'] = dataset.loc[:, 'ConfirmedCases'] / dataset.loc[:, 'Population']



def create_population_per_area(dataset):

    dataset.loc[:, 'PopulationPerArea'] = dataset.loc[:, 'Population'] / dataset.loc[:, 'Area']
def lastNDaysMask(n, days):

    mask = np.zeros((n, n))

    for i in range(days):

        mask += np.eye(n, k=(-1 - i))

    return mask



def buildLastDaysMasks(dataset_length):

    return {2: lastNDaysMask(dataset_length, 2),

                4: lastNDaysMask(dataset_length, 4),

                8: lastNDaysMask(dataset_length, 8),

                16: lastNDaysMask(dataset_length, 16)}



def addGrowthFromLastDaysForProvince(dataset, provinceMask, featureName):

    provinceSubset = dataset.loc[provinceMask]

    lastDaysMasks = buildLastDaysMasks(len(provinceSubset))

    featureValues = provinceSubset.loc[:, featureName].values

    for numDays in [2, 4, 8, 16]:

        lastDaysAvg = ((lastDaysMasks[numDays] * featureValues).sum(axis=1) / numDays)

        dataset.loc[provinceMask, 'Growth' + featureName + 'From' + str(numDays) + 'DaysAvg'] = np.nan_to_num((featureValues / lastDaysAvg), nan=0, posinf=0, neginf=0)



def addGrowthToNextDayForProvince(dataset, provinceMask, featureName):

    provinceSubset = dataset.loc[provinceMask]

    nextDayMask = np.eye(len(provinceSubset), k=1)

    featureValues = provinceSubset.loc[:, featureName].values

    nextDayValues = (nextDayMask * featureValues).sum(axis=1)

    dataset.loc[provinceMask, 'Growth' + featureName + 'ToNextDay'] = np.nan_to_num((nextDayValues / featureValues), nan=0, posinf=0, neginf=0)



def create_growth_features(dataset):

    for (province, country) in dataset[['Province_State', 'Country_Region']].drop_duplicates().values:

        provinceMask = (dataset.Province_State == province) & (dataset.Country_Region == country)

        addGrowthFromLastDaysForProvince(dataset, provinceMask, 'ConfirmedCases')

        addGrowthFromLastDaysForProvince(dataset, provinceMask, 'Fatalities')

        addGrowthToNextDayForProvince(dataset, provinceMask, 'ConfirmedCases')

        addGrowthToNextDayForProvince(dataset, provinceMask, 'Fatalities')
def create_features(dataset):

    create_cases_per_population(dataset)

    create_population_per_area(dataset)

    create_growth_features(dataset)
def extract_defaults_for_missing(dataset):

    defaults_for_missing = {

        'Province_State': '',

        'Population':  dataset.loc[:, 'Population'].quantile(0.1),

        'Area':  dataset.loc[:, 'Area'].quantile(0.1),

        'HealthExpPerCapita': dataset['HealthExpPerCapita'].median(),

        'PhysiciansPer1000': dataset['PhysiciansPer1000'].median(),

        'NursesPer1000': dataset['NursesPer1000'].median()

    }

    return defaults_for_missing
def imput_missing_defaults(dataset, default_features):

    for feature_name in default_features:

        if feature_name in dataset.columns:

            default_value = default_features[feature_name]

            dataset[feature_name].fillna(default_value, inplace=True)

            

def convert_datatypes(dataset):

    if 'Date' in dataset.columns:

        dataset.loc[:, ['Date']] = pd.to_datetime(dataset['Date'])
defaults_for_missing = extract_defaults_for_missing(merged_data)

imput_missing_defaults(merged_data, defaults_for_missing)

convert_datatypes(merged_data)



create_features(merged_data)



merged_data.info()
merged_data.loc[(merged_data.Country_Region=='Brazil')]
MOST_AFFECTED_REGIONS = [(province, state)

                         for (province, state)

                         in merged_data.loc[(merged_data.Date == PUBLIC_LEADERBOARD_USE_REAL_DATA_UP_TO) &

                                            (merged_data.ConfirmedCases >= 0), # was 1000

                                            ['Province_State', 'Country_Region']].values]



def filter_for_training(dataset, last_day_of_available_real_data):

    return dataset.loc[(dataset.Date >= '2020-02-07') &

                       (dataset.Date <= last_day_of_available_real_data) &

                       ([region in MOST_AFFECTED_REGIONS for region in zip(dataset.Province_State, dataset.Country_Region)])]



def features_for_training(targetFeature):

    features = ['HealthExpPerCapita','PhysiciansPer1000', 'NursesPer1000', 'CasesPerPopulation', 'PopulationPerArea']

    for numDays in [2, 4, 8, 16]:

        features = features + ['Growth' + targetFeature + 'From' + str(numDays) + 'DaysAvg']

    return features

    

def growth_feature_to_predict(targetFeature):

    return 'Growth' + targetFeature + 'ToNextDay'



def project_for_training(dataset, targetFeature):

    return dataset.loc[:, features_for_training(targetFeature) + [growth_feature_to_predict(targetFeature)]]



def project_for_predicting(dataset, targetFeature):

    return dataset.loc[:, features_for_training(targetFeature)]
from sklearn.preprocessing import StandardScaler



def extract_data_to_preprocess(dataset):

    features_to_scale = set([item

                             for featureName in ['ConfirmedCases', 'Fatalities']

                             for item in (features_for_training(featureName) + [growth_feature_to_predict(featureName)])])

    scalers = {}

    for feature in features_to_scale:

        scalers[feature] = StandardScaler().fit(dataset[[feature]])

    return (scalers)
merged_data[np.isinf(merged_data.PopulationPerArea)]



(scalers) = extract_data_to_preprocess(merged_data)



for feature_name in scalers:

    print('scaler', feature_name, ', mean:', scalers[feature_name].mean_, ', var:', scalers[feature_name].var_)
def scale_features(dataset, scalers):

    for feature_name in scalers:

        if feature_name in dataset.columns:

            try:

                dataset[[feature_name]] = scalers[feature_name].transform(dataset[[feature_name]])

            except ValueError as e:

                print(dataset)

                raise e

                
def get_preprocessed_dataset(original_dataset, scalers):

    dataset = original_dataset.copy(deep = True)

    scale_features(dataset, scalers)

    return dataset
get_preprocessed_dataset(project_for_training(filter_for_training(merged_data, PUBLIC_LEADERBOARD_USE_REAL_DATA_UP_TO), 'ConfirmedCases'), scalers).sample(10)
get_preprocessed_dataset(project_for_training(filter_for_training(merged_data, PUBLIC_LEADERBOARD_USE_REAL_DATA_UP_TO), 'Fatalities'), scalers).sample(10)
def split_training_and_validation_sets(dataset, validation_percent):

    validation_mask = np.random.rand(len(dataset)) < validation_percent

    training_set = dataset[~validation_mask]

    validation_set = dataset[validation_mask]

    return (training_set, validation_set)
from sklearn import svm



def get_svr_predictor_for_feature(train_dataset, target_column, c_value, gamma_value):

    alg = svm.SVR(gamma=gamma_value, kernel='rbf', C=c_value)

    alg.fit(train_dataset.loc[:, train_dataset.columns != target_column], train_dataset[target_column])

    return alg
import lightgbm as lgb



DEFAULT_PARAMS_CASES = {"bagging_fraction": 0.9,

                        "device": "cpu",

                        "feature_fraction": 0.8,

                        "learning_rate": 0.015,

                        "metric": "rmse",

                        "num_leaves": 4,

                        "min_data_in_leaf": 16,

                        "objective": "regression",

                        "reg_alpha": 0.1,

                        "reg_lambda": 0.1,

                        "seed": SEED}



DEFAULT_PARAMS_FATALITIES = {"bagging_fraction": 0.9,

                             "device": "cpu",

                             "feature_fraction": 0.8,

                             "learning_rate": 0.015,

                             "metric": "rmse",

                             "num_leaves": 4,

                             "min_data_in_leaf": 8,

                             "objective": "regression",

                             "reg_alpha": 0.1,

                             "reg_lambda": 0.1,

                             "seed": SEED}



def get_gbm_predictor_for_feature(train_dataset, target_column, algorithm_params):

    (training_set, validation_set) = split_training_and_validation_sets(train_dataset, 0.2)

    

    if target_column == 'ConfirmedCases':

        params = DEFAULT_PARAMS_CASES.copy()

    else:

        params = DEFAULT_PARAMS_FATALITIES.copy()

        

    for param_name, param_value in algorithm_params.items():

        params[param_name] = param_value

    

    tr = lgb.Dataset(train_dataset.loc[:, train_dataset.columns != target_column], label=train_dataset[target_column])

    

    #tr = lgb.Dataset(training_set.loc[:, training_set.columns != target_column], label=training_set[target_column])

    #val = lgb.Dataset(validation_set.loc[:, validation_set.columns != target_column], label=validation_set[target_column])

    

    bst = lgb.train(params, tr, num_boost_round=400, verbose_eval = -1)

    

    #bst = lgb.train(params, tr, num_boost_round=400, valid_sets=[val], verbose_eval = -1)

        

    return bst
from sklearn.metrics import mean_squared_log_error



def calculate_error(predictions, actual):

    return np.sqrt(mean_squared_log_error(actual, predictions))
def preprocess_and_train_for_feature(dataset, scalers, featureName, algorithm_params):

    complete_data_for_training = project_for_training(dataset, featureName)

    train = get_preprocessed_dataset(complete_data_for_training, scalers)

    

    target_column = growth_feature_to_predict(featureName)

    alg = get_gbm_predictor_for_feature(train, target_column, algorithm_params)

    return alg
def preprocess_and_predict_for_feature(dataset, scalers, featureName, alg):

    complete_data_for_predicting = project_for_predicting(dataset, featureName)

    test = get_preprocessed_dataset(complete_data_for_predicting, scalers)

    

    predicted_growth_scaled = alg.predict(test)

    

    growth_feature_name = growth_feature_to_predict(featureName)

    predicted_growth = scalers[growth_feature_name].inverse_transform(predicted_growth_scaled)

    

    # Predicted growth must be non-negative

    predicted_growth[predicted_growth < 0.0] = 0.0

    

    return predicted_growth
from sklearn import svm



def train_and_predict(target_column, training_set, validation_set, scalers, algorithm_params):

    

    alg = preprocess_and_train_for_feature(training_set, scalers, target_column, algorithm_params)

    

    predictions_train = preprocess_and_predict_for_feature(training_set, scalers, target_column, alg)

    actual_train = training_set.loc[:, growth_feature_to_predict(target_column)]

    error_train = calculate_error(predictions_train, actual_train)

    

    predictions_val = preprocess_and_predict_for_feature(validation_set, scalers, target_column, alg)

    actual_val = validation_set.loc[:, growth_feature_to_predict(target_column)]

    error_val = calculate_error(predictions_val, actual_val)



    return (error_train, error_val)
if TEST_TRAIN_AND_PREDICT:

    filtered_merged_data = filter_for_training(merged_data, PUBLIC_LEADERBOARD_USE_REAL_DATA_UP_TO)



    (error_train_cases, error_val_cases) = train_and_predict('ConfirmedCases', filtered_merged_data, filtered_merged_data, scalers, {"reg_alpha": CASES_REG_ALPHA, "num_leaves": CASES_NUM_LEAVES})

    (error_train_fatalities, error_val_fatalities) = train_and_predict('Fatalities', filtered_merged_data, filtered_merged_data, scalers, {"reg_alpha": FATALITIES_REG_ALPHA, "num_leaves": FATALITIES_NUM_LEAVES})



    print('Training prediction error (ConfirmedCases):', error_train_cases)

    print('Training prediction error (Fatalities):', error_train_fatalities)



#SVR

#Training prediction error (ConfirmedCases): 0.14009188606673692

#Training prediction error (Fatalities): 0.10828247058967111



#GBM

#Training prediction error (ConfirmedCases): 0.10621665813527417

#Training prediction error (Fatalities): 0.10720360641223572
import matplotlib.pyplot as plt



def plot_decreasing_error(target_column, training_set, validation_set, scalers, algorithm_params, subplot):

    plot_steps = 20

    n = len(training_set)

    sample_sizes = list(range(n // plot_steps, n, n // plot_steps))

    

    decreasing_error = [train_and_predict(target_column, training_set[0:size], validation_set, scalers, algorithm_params)

                        for size in sample_sizes]

    

    (_, final_val_error) = decreasing_error[-1]

    

    subplot.plot([error_train for (error_train, _) in decreasing_error], label='training error')

    subplot.plot([error_val for (_, error_val) in decreasing_error], label='validation error')

    subplot.legend(loc='lower right')

    ((param1_name, param1_value), (param2_name, param2_value)) = algorithm_params.items()

    subplot.set_title(param1_name + '=' + str(round(param1_value, 4)) + ', ' + param2_name + '=' + str(round(param2_value, 4)) + ', val_error=' + str(round(final_val_error, 4)))

import matplotlib.pyplot as plt

import itertools



def plot_learning_curves(target_column, dataset, scalers, validation_percent):

    (training_set, validation_set) = split_training_and_validation_sets(dataset, validation_percent)

    

    #params = {"num_leaves":  [4, 8, 16, 32, 64, 128],

    #          "reg_alpha": [0.04, 0.2, 1.0, 5.0, 25.0]}

    

    #params = {"num_leaves":  [4, 8, 16, 32, 64, 128],

    #          "min_data_in_leaf": [4, 8, 16, 32, 64, 128]} #default 20

    

    params = {"feature_fraction":  [1.0, 0.9, 0.8, 0.7, 0.6],

              "learning_rate": [0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025]} #default 0.1

    

    ((param1_name, param1_values), (param2_name, param2_values)) = params.items()



    fig, axs = plt.subplots(len(param1_values), len(param2_values), sharex=True, sharey=True)

    fig.set_size_inches(75, 60)

    

    for i, param1_value in enumerate(param1_values):

        for j, param2_value in enumerate(param2_values):

            #if (i >= 2 or j >= 2):

            #    continue

            param_obj = {param1_name: param1_value, param2_name: param2_value}

            plot_decreasing_error(target_column, training_set, validation_set, scalers, param_obj, subplot=axs[i, j])
if PLOT_LEARNING_CURVES:

    filtered_merged_data = filter_for_training(merged_data, PUBLIC_LEADERBOARD_USE_REAL_DATA_UP_TO)

    plot_learning_curves('ConfirmedCases', filtered_merged_data, scalers, 0.2)
if PLOT_LEARNING_CURVES:

    filtered_merged_data = filter_for_training(merged_data, PUBLIC_LEADERBOARD_USE_REAL_DATA_UP_TO)

    plot_learning_curves('Fatalities', filtered_merged_data, scalers, 0.2)
filtered_data_for_public_leaderboard_prediction = filter_for_training(merged_data, PUBLIC_LEADERBOARD_USE_REAL_DATA_UP_TO)

filtered_data_for_private_leaderboard_prediction = filter_for_training(merged_data, PRIVATE_LEADERBOARD_USE_REAL_DATA_UP_TO)



alg = {}

alg[('ConfirmedCases', PUBLIC_LEADERBOARD_USE_REAL_DATA_UP_TO)

   ] = preprocess_and_train_for_feature(filtered_data_for_public_leaderboard_prediction, scalers, 'ConfirmedCases', {"reg_alpha": CASES_REG_ALPHA, "num_leaves": CASES_NUM_LEAVES})

alg[('Fatalities', PUBLIC_LEADERBOARD_USE_REAL_DATA_UP_TO)

   ] = preprocess_and_train_for_feature(filtered_data_for_public_leaderboard_prediction, scalers, 'Fatalities', {"reg_alpha": FATALITIES_REG_ALPHA, "num_leaves": FATALITIES_NUM_LEAVES})

alg[('ConfirmedCases', PRIVATE_LEADERBOARD_USE_REAL_DATA_UP_TO)

   ] = preprocess_and_train_for_feature(filtered_data_for_private_leaderboard_prediction, scalers, 'ConfirmedCases', {"reg_alpha": CASES_REG_ALPHA, "num_leaves": CASES_NUM_LEAVES})

alg[('Fatalities', PRIVATE_LEADERBOARD_USE_REAL_DATA_UP_TO)

   ] = preprocess_and_train_for_feature(filtered_data_for_private_leaderboard_prediction, scalers, 'Fatalities', {"reg_alpha": FATALITIES_REG_ALPHA, "num_leaves": FATALITIES_NUM_LEAVES})
def build_last_days_masks_for_dataset_of_length_17():

    masks = buildLastDaysMasks(17)

    return {2: masks[2][-1],

                4: masks[4][-1],

                8: masks[8][-1],

                16: masks[16][-1]}
LAST_DAYS_MASKS_LENGTH_17 = build_last_days_masks_for_dataset_of_length_17()
def predict_next_day_cases(row, last_day_of_available_real_data):

    predicted_cases_growth = preprocess_and_predict_for_feature(row, scalers, 'ConfirmedCases', alg[('ConfirmedCases', last_day_of_available_real_data)])

    predicted_fatalities_growth = preprocess_and_predict_for_feature(row, scalers, 'Fatalities', alg[('Fatalities', last_day_of_available_real_data)])

    

    [predicted_next_day_cases] = row.ConfirmedCases * predicted_cases_growth

    [predicted_next_day_fatalities] = row.Fatalities * predicted_fatalities_growth

    

    return (predicted_next_day_cases, predicted_next_day_fatalities)



def calculate_growths_from_last_days_for_province_day(dataset, feature_name):

    value_day = dataset[feature_name].values[-1]

    feature_values = dataset.loc[:, feature_name].values

    

    growths = {}

    for num_days in [2, 4, 8, 16]:

        last_days_avg = ((LAST_DAYS_MASKS_LENGTH_17[num_days] * feature_values).sum() / num_days)

        growths['Growth' + feature_name + 'From' + str(num_days) + 'DaysAvg'] = (0 if last_days_avg == 0 else (value_day / last_days_avg))

    return growths



def generate_predictions_for_province(real_province_data, from_day, to_day, last_day_of_available_real_data):

    days_to_predict = pd.date_range(last_day_of_available_real_data + pd.Timedelta(days=1), to_day)

    

    extended_data = pd.concat([real_province_data.loc[real_province_data.Date <= last_day_of_available_real_data],

                                pd.DataFrame({'Date': days_to_predict})],

                                sort=False)

    

    convert_datatypes(extended_data)

    

    for day in days_to_predict:

        previous_day_row = extended_data[extended_data.Date == (day - pd.Timedelta(days=1))]

        (predicted_cases, predicted_fatalities) = predict_next_day_cases(previous_day_row, last_day_of_available_real_data)

        

        copy_columns = ['Population', 'HealthExpPerCapita', 'PhysiciansPer1000', 'NursesPer1000', 'PopulationPerArea']



        extended_data.loc[extended_data.Date == day, copy_columns] = previous_day_row.loc[:, copy_columns].values

        extended_data.loc[extended_data.Date == day, ['ConfirmedCases', 'Fatalities']] = (predicted_cases, predicted_fatalities)

        extended_data.loc[extended_data.Date == day, 'CasesPerPopulation'] = predicted_cases / extended_data.loc[extended_data.Date == day, 'Population']

        

        dataset_to_calculate_growth_features = extended_data[(extended_data.Date >= (day - pd.Timedelta(days=16))) & (extended_data.Date <= day)]

        

        for feature_name in ['ConfirmedCases', 'Fatalities']:

            for growth_feature_name, value in calculate_growths_from_last_days_for_province_day(dataset_to_calculate_growth_features, feature_name).items():

                extended_data.loc[extended_data.Date == day, growth_feature_name] = value

        

    return extended_data.loc[(extended_data.Date >= from_day) & (extended_data.Date <= to_day), ['ConfirmedCases', 'Fatalities']]



def generate_predictions_for_all_provinces(real_data, predicted_data, from_day, to_day, use_real_data_up_to):

    for (province, country) in real_data[['Province_State', 'Country_Region']].drop_duplicates().values:

        

        if WITH_SINGLE_PROVINCE_COUNTRY and SINGLE_PROVINCE_COUNTRY != (province, country):

            continue

        

        provinceMask = (real_data.Province_State == province) & (real_data.Country_Region == country)

        

        predictions = generate_predictions_for_province(real_data[provinceMask], from_day, to_day, use_real_data_up_to)



        predicted_data.loc[(predicted_data.Province_State == province) &

                           (predicted_data.Country_Region == country) &

                           (pd.to_datetime(predicted_data.Date) >= from_day) &

                           (pd.to_datetime(predicted_data.Date) <= to_day),

                           ['ConfirmedCases', 'Fatalities']] = predictions.values
if GENERATE_SUBMISSION:

    imput_missing_defaults(test_data, defaults_for_missing)

    

    predicted_data = pd.DataFrame(columns=train_data.columns)

    predicted_data = predicted_data.append(test_data.rename(columns={'ForecastId': 'Id'}), sort=False)



    generate_predictions_for_all_provinces(merged_data, predicted_data, PUBLIC_LEADERBOARD_FROM_DAY, PUBLIC_LEADERBOARD_TO_DAY, PUBLIC_LEADERBOARD_USE_REAL_DATA_UP_TO)

    generate_predictions_for_all_provinces(merged_data, predicted_data, PRIVATE_LEADERBOARD_FROM_DAY, PRIVATE_LEADERBOARD_TO_DAY, PRIVATE_LEADERBOARD_USE_REAL_DATA_UP_TO)



    submission_data = pd.DataFrame(columns=['ForecastId', 'ConfirmedCases', 'Fatalities'])

    submission_data = submission_data.append(predicted_data.rename(columns={'Id': 'ForecastId'}).loc[:, ['ForecastId', 'ConfirmedCases', 'Fatalities']], sort=False)

    

    submission_data.loc[submission_data.ConfirmedCases.notnull(), 'ConfirmedCases'] = np.floor(submission_data.loc[submission_data.ConfirmedCases.notnull(), 'ConfirmedCases'])

    submission_data.loc[submission_data.Fatalities.notnull(), 'Fatalities'] = np.floor(submission_data.loc[submission_data.Fatalities.notnull(), 'Fatalities'])



    submission_data.to_csv('submission.csv', index=False)

    print("Your submission was successfully saved!")
merged_data.loc[(merged_data.PopulationPerArea >= 24.783383) & (merged_data.PopulationPerArea <= 24.783385)]



import matplotlib.pyplot as plt

import datetime as dt





def plot_real_and_predicted_feature(real_data, predicted_data, province, country, from_day, to_day, feature_name, subplot):

    



    

    real = real_data[(real_data.Province_State == province) &

                     (real_data.Country_Region == country) #&

                     #(real_data.Date < from_day)

                    ].copy()



    predicted = predicted_data[(predicted_data.Province_State == province) &

                               (predicted_data.Country_Region == country) &

                               (pd.to_datetime(predicted_data.Date) >= from_day) &

                               (pd.to_datetime(predicted_data.Date) <= to_day)].copy()

    

    predicted.loc[:, 'Date'] = pd.to_datetime(predicted.loc[:, 'Date'])

    predicted.loc[:, feature_name] = np.floor(predicted.loc[:, feature_name])

    

    #subplot.set_xticks(ticks={rotation: 90})

    subplot.set_xticklabels( real.Date.dt.strftime('%Y-%m-%d') , rotation=90)

    subplot.plot(real.Date, real[feature_name], label='real')

    subplot.plot(predicted.Date, predicted[feature_name], label='predicted')

    subplot.legend(loc='upper left')

    subplot.set_title(feature_name + ' - ' + ('' if province == '' else province + ' - ') + country)





for (province, country) in merged_data[['Province_State', 'Country_Region']].drop_duplicates().values:

    if WITH_SINGLE_PROVINCE_COUNTRY and SINGLE_PROVINCE_COUNTRY != (province, country):

        continue

    (fig, axs) = plt.subplots(1, 2)

    fig.set_size_inches(12, 4)

    plot_real_and_predicted_feature(merged_data, predicted_data, province, country, PUBLIC_LEADERBOARD_FROM_DAY, PRIVATE_LEADERBOARD_TO_DAY, 'ConfirmedCases', subplot=axs[0])

    plot_real_and_predicted_feature(merged_data, predicted_data, province, country, PUBLIC_LEADERBOARD_FROM_DAY, PRIVATE_LEADERBOARD_TO_DAY, 'Fatalities', subplot=axs[1])



#plot_real_and_predicted_feature(merged_data, predicted_data, '', 'Brazil', PUBLIC_LEADERBOARD_FROM_DAY, PUBLIC_LEADERBOARD_TO_DAY, 'Fatalities')



#plot_real_and_predicted_feature(merged_data, predicted_data, '', 'Brazil', PRIVATE_LEADERBOARD_FROM_DAY, PRIVATE_LEADERBOARD_TO_DAY, 'ConfirmedCases')