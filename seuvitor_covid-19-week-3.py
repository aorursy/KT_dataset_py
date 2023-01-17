TEST_TRAIN_AND_PREDICT = False

PLOT_LEARNING_CURVES = False

GENERATE_SUBMISSION = True



CASES_REG_ALPHA = 5.0

CASES_NUM_LEAVES = 8

FATALITIES_REG_ALPHA = 5.0

FATALITIES_NUM_LEAVES = 4



SEED = 654321



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



PUBLIC_LEADERBOARD_FROM_DAY = pd.Timestamp('2020-03-26')



PUBLIC_LEADERBOARD_TO_DAY = pd.Timestamp('2020-04-08')

PUBLIC_LEADERBOARD_USE_REAL_DATA_UP_TO = pd.Timestamp('2020-03-25')



PRIVATE_LEADERBOARD_FROM_DAY = pd.Timestamp('2020-04-09')

PRIVATE_LEADERBOARD_TO_DAY = pd.Timestamp('2020-05-07')

PRIVATE_LEADERBOARD_USE_REAL_DATA_UP_TO = pd.Timestamp('2020-04-07')

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
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")



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

                                     'continent']),

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
def filter_for_training(dataset):

    return dataset.loc[(dataset.Date >= '2020-02-07') & (dataset.Date <= '2020-04-02')]



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

            dataset[[feature_name]] = scalers[feature_name].transform(dataset[[feature_name]])
def get_preprocessed_dataset(original_dataset, scalers):

    dataset = original_dataset.copy(deep = True)

    scale_features(dataset, scalers)

    return dataset
get_preprocessed_dataset(project_for_training(filter_for_training(merged_data), 'ConfirmedCases'), scalers).sample(10)
get_preprocessed_dataset(project_for_training(filter_for_training(merged_data), 'Fatalities'), scalers).sample(10)
from sklearn import svm



def get_svr_predictor_for_feature(train_dataset, target_column, c_value, gamma_value):

    alg = svm.SVR(gamma=gamma_value, kernel='rbf', C=c_value)

    alg.fit(train_dataset.loc[:, train_dataset.columns != target_column], train_dataset[target_column])

    return alg
import lightgbm as lgb



def get_gbm_predictor_for_feature(train_dataset, target_column, reg_alpha, num_leaves):

    

    (training_set, validation_set) = split_training_and_validation_sets(train_dataset, 0.2)

    

    params = {"bagging_fraction": 0.9,

              "device": "gpu",

              "feature_fraction": 0.8,

              "learning_rate": 0.015,

              "metric": "rmse",

              "num_leaves": num_leaves,

              "objective": "regression",

              "reg_alpha": reg_alpha,

              "reg_lambda": 0.1,

              "seed": SEED}

    

    #tr = lgb.Dataset(train_dataset.loc[:, train_dataset.columns != target_column], label=train_dataset[target_column])

    tr = lgb.Dataset(training_set.loc[:, training_set.columns != target_column], label=training_set[target_column])

    val = lgb.Dataset(validation_set.loc[:, validation_set.columns != target_column], label=validation_set[target_column])

    #bst = lgb.train(params, tr, num_boost_round=200)

    bst = lgb.train(params, tr, num_boost_round=400, valid_sets=[val])

    

    return bst
from sklearn.metrics import mean_squared_log_error



def calculate_error(predictions, actual):

    return np.sqrt(mean_squared_log_error(actual, predictions))
def preprocess_and_train_for_feature(dataset, scalers, featureName, reg_alpha, num_leaves):

    complete_data_for_training = project_for_training(filter_for_training(dataset), featureName)

    train = get_preprocessed_dataset(complete_data_for_training, scalers)

    

    target_column = growth_feature_to_predict(featureName)

    alg = get_gbm_predictor_for_feature(train, target_column, reg_alpha, num_leaves)

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



def train_and_predict(target_column, training_set, validation_set, scalers, reg_alpha, num_leaves):

    

    alg = preprocess_and_train_for_feature(training_set, scalers, target_column, reg_alpha, num_leaves)

    

    predictions_train = preprocess_and_predict_for_feature(training_set, scalers, target_column, alg)

    actual_train = training_set.loc[:, growth_feature_to_predict(target_column)]

    error_train = calculate_error(predictions_train, actual_train)

    

    predictions_val = preprocess_and_predict_for_feature(validation_set, scalers, target_column, alg)

    actual_val = validation_set.loc[:, growth_feature_to_predict(target_column)]

    error_val = calculate_error(predictions_val, actual_val)



    return (error_train, error_val)
if TEST_TRAIN_AND_PREDICT:

    filtered_merged_data = filter_for_training(merged_data)



    (error_train_cases, error_val_cases) = train_and_predict('ConfirmedCases', filtered_merged_data, filtered_merged_data, scalers, CASES_REG_ALPHA, CASES_NUM_LEAVES)

    (error_train_fatalities, error_val_fatalities) = train_and_predict('Fatalities', filtered_merged_data, filtered_merged_data, scalers, FATALITIES_REG_ALPHA, FATALITIES_NUM_LEAVES)



    print('Training prediction error (ConfirmedCases):', error_train_cases)

    print('Training prediction error (Fatalities):', error_train_fatalities)



#SVR

#Training prediction error (ConfirmedCases): 0.14009188606673692

#Training prediction error (Fatalities): 0.10828247058967111



#GBM

#Training prediction error (ConfirmedCases): 0.10621665813527417

#Training prediction error (Fatalities): 0.10720360641223572
def split_training_and_validation_sets(dataset, validation_percent):

    validation_mask = np.random.rand(len(dataset)) < validation_percent

    training_set = dataset[~validation_mask]

    validation_set = dataset[validation_mask]

    return (training_set, validation_set)
import matplotlib.pyplot as plt



def plot_decreasing_error(target_column, training_set, validation_set, scalers, reg_alpha, num_leaves, subplot):

    plot_steps = 20

    n = len(training_set)

    sample_sizes = list(range(n // plot_steps, n, n // plot_steps))

    

    decreasing_error = [train_and_predict(target_column, training_set[0:size], validation_set, scalers, reg_alpha, num_leaves)

                        for size in sample_sizes]

    

    (_, final_val_error) = decreasing_error[-1]

    

    subplot.plot([error_train for (error_train, _) in decreasing_error], label='training error')

    subplot.plot([error_val for (_, error_val) in decreasing_error], label='validation error')

    subplot.legend(loc='lower right')

    subplot.set_title('num_leaves=' + str(round(gamma_value, 4)) + ', reg_alpha=' + str(round(c_value, 4)) + ', val_error=' + str(round(final_val_error, 4)))
import matplotlib.pyplot as plt



def plot_learning_curves(target_column, dataset, scalers, validation_percent):

    (training_set, validation_set) = split_training_and_validation_sets(dataset, validation_percent)

    

    num_leaves_values = [4, 8, 16, 32, 64, 128] # num_leaves

    reg_alpha_values = [0.04, 0.2, 1.0, 5.0, 25.0] # reg_alpha

    

    fig, axs = plt.subplots(len(reg_alpha_values), len(num_leaves_values), sharex=True, sharey=True)

    fig.set_size_inches(75, 60)



    for i in range(len(reg_alpha_values)):

        for j in range(len(num_leaves_values)):

            plot_decreasing_error(target_column, training_set, validation_set, scalers, reg_alpha=reg_alpha_values[i], num_leaves=num_leaves_values[j], subplot=axs[i, j])
if PLOT_LEARNING_CURVES:

    filtered_merged_data = filter_for_training(merged_data)

    plot_learning_curves('ConfirmedCases', filtered_merged_data, scalers, 0.2)
if PLOT_LEARNING_CURVES:

    filtered_merged_data = filter_for_training(merged_data)

    plot_learning_curves('Fatalities', filtered_merged_data, scalers, 0.2)
alg = {}

alg['ConfirmedCases'] = preprocess_and_train_for_feature(merged_data, scalers, 'ConfirmedCases', CASES_REG_ALPHA, CASES_NUM_LEAVES)

alg['Fatalities'] = preprocess_and_train_for_feature(merged_data, scalers, 'Fatalities', FATALITIES_REG_ALPHA, FATALITIES_NUM_LEAVES)
def build_last_days_masks_for_dataset_of_length_17():

    masks = buildLastDaysMasks(17)

    return {2: masks[2][-1],

                4: masks[4][-1],

                8: masks[8][-1],

                16: masks[16][-1]}
LAST_DAYS_MASKS_LENGTH_17 = build_last_days_masks_for_dataset_of_length_17()
def predict_next_day_cases(row):

    predicted_cases_growth = preprocess_and_predict_for_feature(row, scalers, 'ConfirmedCases', alg['ConfirmedCases'])

    predicted_fatalities_growth = preprocess_and_predict_for_feature(row, scalers, 'Fatalities', alg['Fatalities'])

    

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

        (predicted_cases, predicted_fatalities) = predict_next_day_cases(previous_day_row)

        

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

        provinceMask = (real_data.Province_State == province) & (real_data.Country_Region == country)

        

        predictions = generate_predictions_for_province(real_data[provinceMask], from_day, to_day, use_real_data_up_to)



        predicted_data.loc[(predicted_data.Province_State == province) &

                           (predicted_data.Country_Region == country) &

                           (pd.to_datetime(predicted_data.Date) >= from_day) &

                           (pd.to_datetime(predicted_data.Date) <= to_day), ['ConfirmedCases', 'Fatalities']] = predictions.values
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