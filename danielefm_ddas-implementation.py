import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

import random
import time

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)
pd.options.mode.chained_assignment = None
init_notebook = time.time()
fdums_household = pd.read_csv('../input/urban-mobility-survey-federal-district-brazil/Household.csv', sep=';')
fdums_person = pd.read_csv('../input/urban-mobility-survey-federal-district-brazil/Person.csv', sep=';')
fdums_trip = pd.read_csv('../input/urban-mobility-survey-federal-district-brazil/Trip.csv', sep=';')
fdums_person.columns
fdums_household.columns
fdums_trip.columns
soc_columns = ['person_id', 'household_id', 'age', 'gender', 'area_of_occupation', 'education_level', 'has_driver_license']
soc = fdums_person[soc_columns]

print('The number of rows in the soc dataframe is '
      + str(len(soc)))

soc.head()
import_columns = ['household_id','people_in_household','vehicles']
household_info = fdums_household[import_columns]
household_info.head()
soc = pd.merge(soc, household_info, on='household_id')
soc.head()
individuals = fdums_trip.person_id.unique()
soc = soc.loc[soc['person_id'].isin(individuals)]

print('The number of rows in the soc dataframe is '
      + str(len(soc)))

soc.head()
print('The number of rows in the soc dataframe with missing values is '
      + str(soc.shape[0] - soc.dropna().shape[0]))
soc = soc.dropna()
print('The number of rows in the soc dataframe is '
      + str(len(soc)))
soc.age.unique()
map_age = {'0 to 4 years old': 1,
           '5 to 9 years old': 2,
           '10 to 14 years old': 3,
           '15 to 17 years old': 4,
           '18 to 19 years old': 5,
           '20 to 24 years old': 6,
           '25 to 29 years old': 7,
           '30 to 39 years old': 8,
           '40 to 49 years old': 9,
           '50 to 59 years old': 10,
           '60 to 69 years old': 11,
           '70 to 79 years old': 12,
           'More than 80 years old': 13}

soc.age = soc.age.map(map_age)

soc.rename(columns={'age':'age_group'}, inplace=True)

soc.head()
    
soc.education_level.unique()
map_education = {'Younger than 6 years old / not student': 1,
                 'Illiterate': 1,
                 'Literate, but with no education level': 2,
                 'Early childhood/kindergarden level': 3,
                 'Incomplete lower-secondary/middle school': 4,
                 'Complete lower-secondary/middle school': 5,
                 'Incomplete upper-secondary/high school': 6,
                 'Complete upper-secondary/high school': 7, 
                 'Incomplete undergraduate school': 8,
                 'Complete undergraduate school': 9,
                 'Complete graduate school': 10}

soc.education_level = soc.education_level.map(map_education)
soc.head()
map_gender = {'Male':0,
              'Female':1}

map_has_driver_license = {'No':0,
                          'Yes':1}

soc.gender = soc.gender.map(map_gender)
soc.has_driver_license = soc.has_driver_license.map(map_has_driver_license)

soc.rename(columns={'gender':'is_female'}, inplace=True)

soc.head()
soc.area_of_occupation.unique()
soc = soc.rename(columns={'area_of_occupation': 'is_student'})

map_is_student = {'Retired': 0,
                  'Other': 0,
                  'Self-employed (professional)': 0,
                  'Student (regular courses)': 1,
                  'Homekeeper': 0,
                  'Businessperson': 0,
                  'Private worker': 0,
                  'Civil servent': 0,
                  'No activity': 0,
                  'Unemployed': 0,
                  'Domestic worker': 0,
                  'Self-employed (casual)': 0,
                  'Student (extension courses)': 1,
                  'Volunteer work':0}

soc.is_student = soc.is_student.map(map_is_student)

soc.head()
soc = soc.rename(columns={'vehicles': 'is_car_available'})

def map_is_car_available(row):
    if row.is_car_available>=1:
        row.is_car_available = 1
    else:
        row.is_car_available = 0
    return row

soc = soc.apply(map_is_car_available, axis='columns')

soc.head()
soc = soc.drop(soc[soc.age_group <= 3].index)
print('The number of rows in the soc dataframe is '+str(len(soc)))
fdums_trip.columns
import_columns = ['person_id',
                  'trip_id',
                  'ar_origin',
                  'ar_destination',
                  'activity_origin',
                  'activity_destination',
                  'includes_walk',
                  'includes_bicycle',
                  'includes_subway',
                  'includes_brt',
                  'includes_bus',
                  'includes_unlicensed_service',
                  'includes_private_charter',
                  'includes_school_bus',
                  'includes_car_as_driver',
                  'includes_car_as_passenger',
                  'includes_motorcycle_as_driver',
                  'includes_motorcycle_as_passenger',
                  'includes_taxi',
                  'includes_motorbicycle_taxi',
                  'includes_private_driver',
                  'includes_other_modes']

trips = fdums_trip[import_columns]

print('The trips dataframe contains information about '
      + str(len(trips))
      + ' trips performed by '
      + str(len(trips.person_id.unique()))
      + ' people')

trips.head()
person_missing_values = trips[pd.isnull(trips.activity_destination)].person_id
trips = trips.drop(trips[trips.person_id.isin(person_missing_values)].index)

print('The trips dataframe contains information about '
      + str(len(trips))
      + ' trips performed by '
      + str(len(trips.person_id.unique()))
      + ' people')

trips.head()
# Slow cell
incomplete_diaries = trips.copy()
incomplete_diaries = incomplete_diaries[incomplete_diaries.groupby('person_id').person_id.transform(len) == 1]

print('There are '
      + str(len(incomplete_diaries))
      + ' people with incomplete diaries within the trips dataset')

incomplete_diaries.head()
person_incomplete_diary = incomplete_diaries.person_id
trips = trips.drop(trips[trips.person_id.isin(person_incomplete_diary)].index)

print('The trips dataframe contains information about '
      + str(len(trips))
      + ' trips performed by '
      + str(len(trips.person_id.unique()))
      + ' people')
trips.activity_destination.unique()
map_activity_type = {'Shop':'shop',
                     'Home':'sleep', 
                     'Main workplace':'work', 
                     'Main study place': 'school',
                     'Taking someone somewhere':'other',
                     'Other':'other', 
                     'Eating out':'shop',
                     'Secondary study place':'school',
                     'Personal matters':'other', 
                     'Leisure':'leisure',
                     'Secondary workplace':'work', 
                     'Health':'other', 
                     'Business':'work'}

trips.activity_origin = trips.activity_origin.map(map_activity_type)
trips.activity_destination = trips.activity_destination.map(map_activity_type)

trips.activity_origin.unique()
def compute_main_activ_places(input_trips):
    '''Returns a dictionary with the Administrative Region where
    each person performs their main activity.
    
    Keyword arguments:
    input_trips -- dataframe containing trip information
    '''
    trips = input_trips.copy()

    dict_activity = dict.fromkeys(trips.person_id.unique())
    dict_places = dict.fromkeys(trips.person_id.unique())
    
    for row in trips.to_numpy():
        person_id = row[0]
        activity = row[5]
        cur_main_activity = dict_activity[person_id]
        
        if activity == 'work':
            if cur_main_activity == 'work':
                pass
            else:
                dict_activity[person_id]='work'
                dict_places[person_id]=row[3]
        
        elif activity =='school':
            if cur_main_activity == 'work':
                pass
            elif cur_main_activity == 'school':
                pass
            else:
                dict_activity[person_id]='school'
                dict_places[person_id]=row[3]     
        
        else:
            if cur_main_activity == 'work':
                pass
            elif cur_main_activity == 'school':
                pass
            elif cur_main_activity == 'other':
                pass
            else:
                dict_activity[person_id]='other'
                dict_places[person_id]=row[3]
        
    return dict_places

main_places = compute_main_activ_places(trips)

# Displaying the first 5 entries of the dictionary:
dict(list(main_places.items())[0:5])
reach = pd.DataFrame.from_dict(main_places, orient='index', columns=['main_actv_ar'])
reach.reset_index(inplace=True)
reach.rename(columns={'index':'person_id'}, inplace=True)
reach.head()
def compute_home_place(input_person, input_household):
    '''Returns a dictionary with the Administrative Region where
    each person lives.
    
    Keyword arguments:
    input_person -- dataframe with information about each person
    input_household -- dataframe with information about each household
    '''
    person = input_person.copy()
    household = input_household.copy()

    dict_home_ar = dict.fromkeys(person.person_id.unique())
    household.set_index('household_id', inplace=True)
    
    for row in person.to_numpy():
        this_person = row[0]
        this_household = row[1]
        dict_home_ar[this_person] = household.loc[this_household, 'administrative_region']
    
    return dict_home_ar

home_ar = compute_home_place(fdums_person, fdums_household)

# Displaying the first 5 entries of the dictionary:
dict(list(home_ar.items())[0:5])
reach['home_ar'] = reach['person_id'].map(home_ar)
reach = reach[['person_id', 'home_ar', 'main_actv_ar']]
reach.head()
bike_distance_matrix = pd.read_csv('../input/distance-matrix-distrito-federal/reach_bicycling.csv', sep=';')

# Displaying the first five rows and columns
bike_distance_matrix.iloc[:5,:5]
bike_distance_matrix.rename(columns={'Unnamed: 0':'origin'}, inplace=True)
bike_distance_matrix.set_index('origin', inplace=True)

print('This Distance Matrix has '
      + str(len(bike_distance_matrix)) 
      + ' rows (possible origins) and ' 
      + str(len(bike_distance_matrix.columns))
      + ' columns (possible destinations)')

# Displaying the first five rows and columns
bike_distance_matrix.iloc[:5,:5]
car_distance_matrix = pd.read_csv('../input/distance-matrix-distrito-federal/reach_driving.csv', sep=';')
car_distance_matrix.rename(columns={'Unnamed: 0':'origin'}, inplace=True)
car_distance_matrix.set_index('origin', inplace=True)

transit_distance_matrix = pd.read_csv('../input/distance-matrix-distrito-federal/reach_transit.csv', sep=';')
transit_distance_matrix.rename(columns={'Unnamed: 0':'origin'}, inplace=True)
transit_distance_matrix.set_index('origin', inplace=True)

walk_distance_matrix = pd.read_csv('../input/distance-matrix-distrito-federal/reach_walking.csv', sep=';')
walk_distance_matrix.rename(columns={'Unnamed: 0':'origin'}, inplace=True)
walk_distance_matrix.set_index('origin', inplace=True)
def reach_from_distance_matrix(input_person, input_distance_matrix):
    '''Returns a dictionary with reach information (average trip duration
    between the person's Administrative Region (AR) and the AR where
    he/she performs their main activity, for a given mode).
    
    Keyword arguments:
    input_person -- dataframe with information about the person
    input_distance_matrix -- dataframe with the distance matrix for a given mode
    '''
    person = input_person.copy()
    distance_matrix = input_distance_matrix.copy()

    dict_reach = dict.fromkeys(person.person_id.unique())
    
    for row in person.to_numpy():
        this_person = row[0]
        origin = row[1]
        destination = row[2]
       
        try:
            dict_reach[this_person] = distance_matrix.loc[origin, destination]   
        except:
            dict_reach[this_person] = -1
    
    return dict_reach

bike_reach_dict = reach_from_distance_matrix(reach, bike_distance_matrix)

# Displaying the first 5 entries of the dictionary:
dict(list(bike_reach_dict.items())[0:5])
car_reach_dict = reach_from_distance_matrix(reach, car_distance_matrix)
transit_reach_dict = reach_from_distance_matrix(reach, transit_distance_matrix)
walk_reach_dict = reach_from_distance_matrix(reach, walk_distance_matrix)
reach['reach_bike'] = reach['person_id'].map(bike_reach_dict)
reach['reach_car'] = reach['person_id'].map(car_reach_dict)
reach['reach_transit'] = reach['person_id'].map(transit_reach_dict)
reach['reach_walk'] = reach['person_id'].map(walk_reach_dict)
reach.head()
reach = reach.drop(['home_ar','main_actv_ar'], axis=1)
reach.head()
soc_and_reach = pd.merge(soc, reach, on='person_id')
soc_and_reach.head()
trips.head()
trips.columns
trips.drop(columns=['ar_origin', 'ar_destination'], inplace=True)
# Slow cell
def create_pt_column(row):
    if (row.includes_subway==1 or row.includes_brt==1 or row.includes_bus==1):
        row.includes_pt = 1
    else:
        row.includes_pt = 0
    
    return row

# Create a new empty column in the trips dataframe
trips['includes_pt'] = ''

# Check if any subtype of the public transportation mode is included in the trip
# for each row
trips = trips.apply(create_pt_column, axis='columns')

# Drop the subtype columns
trips.drop(columns=['includes_subway', 'includes_brt', 'includes_bus'],
           inplace=True)

trips.head()
# Slow cell
def create_car_column(row):
    if (row.includes_car_as_passenger==1 or row.includes_car_as_driver==1):
        row.includes_car = 1
    else:
        row.includes_car = 0
    
    return row

# Create a new empty column in the trips dataframe
trips['includes_car'] = ''

# Check if any subtype of the car mode is included in the trip
# for each row
trips = trips.apply(create_car_column, axis='columns')

# Drop the subtype columns
trips.drop(columns=['includes_car_as_passenger', 'includes_car_as_driver'],
           inplace=True)

trips.head()
# Slow cell
def create_other_column(row):
    if (row.includes_unlicensed_service==1 or
        row.includes_private_charter==1 or
        row.includes_school_bus==1 or
        row.includes_motorcycle_as_driver==1 or
        row.includes_motorcycle_as_passenger==1 or
        row.includes_taxi==1 or
        row.includes_motorbicycle_taxi==1 or
        row.includes_private_driver==1 or
        row.includes_other_modes==1):
        row.includes_other_modes = 1
    else:
        row.includes_other_modes = 0
    
    return row

# Check if any subtype of other modes is included in the trip
# for each row
trips = trips.apply(create_other_column, axis='columns')

# Drop the subtype columns
trips.drop(columns=['includes_unlicensed_service', 
                    'includes_private_charter', 
                    'includes_school_bus', 
                    'includes_motorcycle_as_driver', 
                    'includes_motorcycle_as_passenger',
                    'includes_taxi',
                    'includes_motorbicycle_taxi',
                    'includes_private_driver'], inplace=True)
trips.head()
print('The trips dataframe contains information about '
      + str(len(trips))
      + ' trips performed by '
      + str(len(trips.person_id.unique()))
      + ' people')

trips.columns
transfer_trips = trips.loc[trips.loc[:,['includes_walk',
                                        'includes_bicycle',
                                        'includes_other_modes',
                                        'includes_pt',
                                        'includes_car']].sum(axis=1)>1]
print('In the trips dataframe, '
      + str(len(transfer_trips.person_id.unique()))
      + ' use more than one transport mode on the same trip')
trips = trips.loc[~trips['person_id'].isin(transfer_trips.person_id)]

print('The trips dataframe contains information about '
      + str(len(trips))
      + ' trips performed by '
      + str(len(trips.person_id.unique()))
      + ' people')

trips.head()
# Slow cell
def create_mode_column(row):
    if (row.includes_car == 1):
        row.mode_type = 'car'
    elif (row.includes_pt == 1):
        row.mode_type = 'pt'
    elif (row.includes_bicycle) == 1:
        row.mode_type = 'bike'
    elif (row.includes_other_modes) == 1:
        row.mode_type = 'other_mode'
    else:
        row.mode_type = 'walk'
    
    return row

# Create a new empty column in the trips dataframe
trips['mode_type'] = ''

# Check which of the mode columns has the value of 1 and
# get the name of that mode
trips = trips.apply(create_mode_column, axis='columns')

# Drop the subtype columns
trips.drop(columns=['includes_walk',
                    'includes_bicycle',
                    'includes_other_modes',
                    'includes_pt',
                    'includes_car'], inplace=True)
trips.head()
def set_last_actv_to_none(input_df):
    '''Returns a dataframe in which the last activity performed
    by each person is set as 'none'
    
    Keyword arguments:
    input_df -- dataframe containing trip information
    '''
    df = input_df.copy()
    
    df['next_person'] = df.person_id.shift(-1)
    df['activity_destination'].loc[(df['person_id'] != df['next_person'])] = 'none'
    df.drop(columns=['next_person'], inplace=True)
    return df

trips = set_last_actv_to_none(trips)
trips.head(13)
def create_actv_counts(input_df, input_count):
    '''Returns a dataframe with activity counts columns, which
    represent the number of times each activity has been performed
    up to that point of the trip
    
    Keyword arguments:
    input_df -- dataframe containing trip information
    input_count -- OHE dataframe of the activities performed, obtained
                   from the input_df
    '''
    df = input_df.copy()
    count = input_count.copy()
    
    # Transform dataframes into lists of lists for speed performance
    df_list = df.to_numpy()
    count_list = count.to_numpy()
    
    # If the person_id of the current row is the same of the one on
    # the previous row (if we're dealing with the same activity diary
    # for a certain person) we must sum up the activities that have
    # been performed up to now
    
    for i in range(1,len(df_list)):
        if (df_list[i,0] == df_list[i-1,0]):
            count_list[i] = count_list[i] + count_list[i-1]
        else:
            pass
    
    # From the list, we create an output dataframe
    output_df = pd.DataFrame(count_list, columns=count.columns)
    
    # We set new indexes for both de the original dataframe and
    # the counts dataframe, so they can be concatenated
    df = df.reindex(range(0,len(df)))
    output_df = output_df.reindex(range(0,len(output_df)))
    
    # The output is the concatenated df
    output_df = pd.concat((df, output_df), axis=1)
    return output_df

# We initiate the activity counts dataframe as the 'one-hot encoded'
# version of the 'activity_origin' column
actv_count = pd.get_dummies(trips['activity_origin'])

trips = create_actv_counts(trips, actv_count)
trips.head()
trips.columns = ['person_id', 'trip_id', 'activity_origin',
                 'activity_destination', 'mode_type', 'count_leisure',
                 'count_other', 'count_school', 'count_shop',
                 'count_sleep', 'count_work']
trips.head()
def create_mode_counts(input_df, input_count):
    '''Returns a dataframe with mode counts columns, which
    represent the number of times each trasport mode has been
    used up to that point of the trip
    
    Keyword arguments:
    input_df -- dataframe containing trip information
    input_count -- OHE dataframe of the mode choice, obtained
                   from the input_df
    '''
    df = input_df.copy()
    count = input_count.copy()
    
    # The 'count' columns have to be shifted by because we want the
    # counts for the previous trips, not including the current one
    count = count.shift(1)
    
    # Transform dataframes into lists of lists for speed performance
    df_list = df.to_numpy()
    count_list = count.to_numpy()
    
    # The first row of the count_list is initiated empty because the
    # first person has not performed any trip yet
    count_list[0] = [0,0,0,0,0]
    
    for i in range(1,len(df_list)):
        if (df_list[i,0] != df_list[i-1,0]):
            count_list[i] = [0,0,0,0,0]
        else:
            count_list[i] = count_list[i] + count_list[i-1]
    
    output_df = pd.DataFrame(count_list, columns=count.columns)
    
    df = df.reindex(range(0,len(df)))
    output_df = output_df.reindex(range(0,len(output_df)))
                           
    output_df = pd.concat((df, output_df), axis=1)
    return output_df

# We initiate the activity counts dataframe as the 'one-hot encoded'
# version of the 'mode_type' column
mode_count = pd.get_dummies(trips['mode_type'])
trips = create_mode_counts(trips, mode_count)
trips.head()
trips.columns = ['person_id', 'trip_id', 'activity_origin', 'activity_destination',
                 'mode_type', 'count_leisure', 'count_other', 'count_school',
                 'count_shop', 'count_sleep', 'count_work',
                 'count_bike', 'count_car',
                 'count_other_mode', 'count_pt', 'count_walk']
trips.head()
trips = pd.concat([trips, pd.get_dummies(trips['activity_origin'], prefix='ohe_origin')], axis=1)
trips = pd.concat([trips, pd.get_dummies(trips['activity_destination'], prefix='ohe_destin')], axis=1)
trips.drop('activity_origin', axis=1, inplace=True)
trips.head()
input_df = pd.merge(soc_and_reach, trips, how='inner', on='person_id')
input_df.drop(['household_id'], axis=1, inplace=True)
input_df.set_index('person_id', inplace=True)

print('The organized input dataframe contains information about '
      + str(len(input_df.index.unique().values))
      + ' individuals and ' 
      + str(len(input_df)) 
      + ' trips')

input_df.head()
input_df.to_csv("input_df.csv")
index_list = input_df.index.unique().values
random.Random(123).shuffle(index_list)

# The element in which the test and training sets are separated 
# is the one on the first fifth of the list length
separation_element = len(index_list)//5

test_index = index_list[:separation_element]
train_index = index_list[separation_element:]
# Because of the random selection, the trips were out of order,
# so we have to sort them again
test_df = input_df.loc[test_index]
test_df.sort_values(by=['trip_id'], inplace=True)

train_df = input_df.loc[train_index]
train_df.sort_values(by=['trip_id'], inplace=True)
ATM_X_columns = ['age_group',
                 'is_female',
                 'is_student',
                 'education_level',
                 'has_driver_license',
                 'people_in_household', 
                 'is_car_available',
                 'reach_bike',
                 'reach_car',
                 'reach_transit',
                 'reach_walk', 
                 'count_leisure', 
                 'count_other',
                 'count_school', 
                 'count_shop',
                 'count_sleep',
                 'count_work',
                 'ohe_origin_leisure',
                 'ohe_origin_other', 
                 'ohe_origin_school', 
                 'ohe_origin_shop',
                 'ohe_origin_sleep', 
                 'ohe_origin_work']

ATM_Y_columns = ['activity_destination']

X_train_ATM = train_df[ATM_X_columns]
Y_train_ATM = train_df[ATM_Y_columns]
MCM_X_columns = ['age_group',
                 'is_female',
                 'is_student',
                 'education_level',
                 'has_driver_license',
                 'people_in_household', 
                 'is_car_available',
                 'reach_bike',
                 'reach_car',
                 'reach_transit',
                 'reach_walk', 
                 'count_leisure', 
                 'count_other',
                 'count_school', 
                 'count_shop',
                 'count_sleep',
                 'count_work',
                 'ohe_origin_leisure',
                 'ohe_origin_other', 
                 'ohe_origin_school', 
                 'ohe_origin_shop',
                 'ohe_origin_sleep', 
                 'ohe_origin_work',
                 'ohe_destin_leisure',
                 'ohe_destin_none', 
                 'ohe_destin_other', 
                 'ohe_destin_school',
                 'ohe_destin_shop',
                 'ohe_destin_sleep',
                 'ohe_destin_work',
                 'count_bike',
                 'count_car',
                 'count_other_mode',
                 'count_pt', 
                 'count_walk']

MCM_Y_columns = ['mode_type']

X_train_MCM = train_df[MCM_X_columns]
Y_train_MCM = train_df[MCM_Y_columns]
def run_cross_validation_on_trees(X, y, tree_depths, cv=5, scoring='f1_micro'):
    """ A function to perform the cross-validation procedure
    for selecting the best specification of the model
    
    Keyword arguments:
    X -- features used as input
    y -- features that has to be predicted
    tree_depths -- list of integers to be used as tree depths
                   in cross-validation
    cv -- number of folds (default = 5)
    scoring -- scoring function to be used as evaluation metric
    """
    
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    
    for depth in tree_depths:
        tree_model = DecisionTreeClassifier(max_depth=depth, random_state=123)
        cv_scores = cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, y).score(X, y))
        
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    
    return cv_scores_mean, cv_scores_std, accuracy_scores
def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title, score):
    
    """Returns a plot with the results from cross validation 
    """
    
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation score', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.plot(depths, accuracy_scores, '-*', label='train score', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel(score, fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()
depths_list = range(1,51)

init_time = time.time()
ATM_dt_cv_scores_mean, ATM_dt_cv_scores_std, ATM_dt_accuracy_scores = run_cross_validation_on_trees(X_train_ATM,
                                                                                                    Y_train_ATM,
                                                                                                    depths_list)
end_time = time.time()
ellapsed_time = end_time - init_time

print('Cross-validation took ' + str(ellapsed_time) + ' seconds to run')

#plotting accuracy
plot_cross_validation_on_trees(depths_list, 
                               ATM_dt_cv_scores_mean, 
                               ATM_dt_cv_scores_std, 
                               ATM_dt_accuracy_scores, 
                               'F1 score per decision tree depth on ATM cross-validation data',
                               'f1-score')
max_f1 = np.amax(np.around(ATM_dt_cv_scores_mean,3))

optimal_depth_ATM = np.argmax(np.around(ATM_dt_cv_scores_mean,3)) + 1

print('The maximum F1 score for the ATM model is '
      + str(max_f1)
      + ' for tree depth '
      + str(optimal_depth_ATM))
X_train_cv_ATM, X_val_ATM, Y_train_cv_ATM, Y_val_ATM = train_test_split(X_train_ATM,
                                                                        Y_train_ATM,
                                                                        test_size = 0.20,
                                                                        random_state=123)

ATM_dt_cv_model = DecisionTreeClassifier(max_depth=optimal_depth_ATM, random_state=123)
ATM_dt_cv_model.fit(X_train_cv_ATM, Y_train_cv_ATM)

Y_pred_val_ATM = ATM_dt_cv_model.predict(X_val_ATM)

class_names = ['leisure', 'none', 'other', 'school', 'shop', 'sleep', 'work']

disp = plot_confusion_matrix(ATM_dt_cv_model, X_val_ATM, Y_val_ATM,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
disp.ax_.set_title('ATM confusion matrix')
plt.show()

print(classification_report(Y_val_ATM, Y_pred_val_ATM, digits=3, labels=class_names))
init_time = time.time()
MCM_dt_cv_scores_mean, MCM_dt_cv_scores_std, MCM_dt_accuracy_scores = run_cross_validation_on_trees(X_train_MCM,
                                                                                                    Y_train_MCM,
                                                                                                    depths_list)
end_time = time.time()
ellapsed_time = end_time - init_time

print('Cross-validation took ' + str(ellapsed_time) + ' seconds to run')

# plotting accuracy
plot_cross_validation_on_trees(depths_list, 
                               MCM_dt_cv_scores_mean, 
                               MCM_dt_cv_scores_std, 
                               MCM_dt_accuracy_scores, 
                               'F1 score per decision tree depth on MCM cross-validation data',
                               'f1-score')
max_f1 = np.amax(np.around(MCM_dt_cv_scores_mean,3))

optimal_depth_MCM = np.argmax(np.around(MCM_dt_cv_scores_mean,3)) + 1

print('The maximum F1 score for the MCM model is '
      + str(max_f1)
      + ' for tree depth '
      + str(optimal_depth_MCM))
X_train_cv_MCM, X_val_MCM, Y_train_cv_MCM, Y_val_MCM = train_test_split(X_train_MCM,
                                                                        Y_train_MCM,
                                                                        test_size = 0.20,
                                                                        random_state=123)

MCM_dt_cv_model = DecisionTreeClassifier(max_depth=optimal_depth_MCM, random_state=123)
MCM_dt_cv_model.fit(X_train_cv_MCM, Y_train_cv_MCM)

Y_pred_val_MCM = MCM_dt_cv_model.predict(X_val_MCM)

class_names = ['bike', 'car','other_mode', 'pt', 'walk']

disp = plot_confusion_matrix(MCM_dt_cv_model, X_val_MCM, Y_val_MCM,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
disp.ax_.set_title('MCM confusion matrix, without normalization')
plt.show()

print(classification_report(Y_val_MCM, Y_pred_val_MCM, digits=3, labels=class_names))
init_time = time.time()

ATM_dt_model = DecisionTreeClassifier(max_depth=optimal_depth_ATM, random_state=123)
ATM_dt_model.fit(X_train_ATM, Y_train_ATM)

MCM_dt_model = DecisionTreeClassifier(max_depth=optimal_depth_MCM, random_state=123)
MCM_dt_model.fit(X_train_MCM, Y_train_MCM)

end_time = time.time()
ellapsed_time = end_time - init_time

print('Model training for both the ATM and MCM took ' + str(ellapsed_time) + ' seconds to run')
X_test_columns = ['age_group',
                  'is_female',
                  'is_student',
                  'education_level',
                  'has_driver_license',
                  'people_in_household', 
                  'is_car_available',
                  'reach_bike',
                  'reach_car',
                  'reach_transit',
                  'reach_walk']

X_test = test_df[X_test_columns].copy()

# We have to drop duplicate columns because the same person
# appears on multiple rows, as the original dataset contained
# all trips that person performed
X_test.reset_index(inplace=True)
X_test.drop_duplicates(inplace=True)
X_test.set_index('person_id', inplace=True)
X_test.head()
class Person:
    def __init__(self, soc_features):
        self.id = soc_features.index # Series type
        self.soc_features = soc_features
        
        self.origin_act_type = 'sleep' # String
        self.origin_act_type_ohe = pd.DataFrame(data={'ohe_origin_leisure':[0],
                                                      'ohe_origin_other':[0],
                                                      'ohe_origin_school':[0],
                                                      'ohe_origin_shop':[0],
                                                      'ohe_origin_sleep':[1],
                                                      'ohe_origin_work':[0]},
                                                index=self.id)

        self.destination_act_type = '' # String
        self.destination_act_type_ohe = pd.DataFrame(data={'ohe_destin_leisure':[0],
                                                           'ohe_destin_other':[0],
                                                           'ohe_destin_none':[0],
                                                           'ohe_destin_school':[0],
                                                           'ohe_destin_shop':[0],
                                                           'ohe_destin_sleep':[0],
                                                           'ohe_destin_work':[0]},
                                                     index=self.id)
        
        self.act_type_counts = pd.DataFrame(data={'count_leisure':[0],
                                                  'count_other':[0],
                                                  'count_school':[0],
                                                  'count_shop':[0],
                                                  'count_sleep':[1],
                                                  'count_work':[0]},
                                            index=self.id)
        
        self.mode_counts = pd.DataFrame(data={'count_bike':[0],
                                              'count_car':[0],
                                              'count_other_mode':[0],
                                              'count_pt':[0],
                                              'count_walk':[0]},
                                       index=self.id)
        
        self.cur_mode = '' # String
        
        self.total_trips = 0
    
    def update_act_counts(self):
        if self.destination_act_type == 'none':
            pass

        else:
            self.act_type_counts.loc[:,'count_'+self.destination_act_type] = self.act_type_counts.loc[:,'count_'+self.destination_act_type] + 1
    
    def update_destination_act_type_ohe(self):
        self.destination_act_type_ohe = pd.DataFrame(data={'ohe_destin_leisure':[0],
                                                           'ohe_destin_none':[0],
                                                           'ohe_destin_other':[0],
                                                           'ohe_destin_school':[0],
                                                           'ohe_destin_shop':[0],
                                                           'ohe_destin_sleep':[0],
                                                           'ohe_destin_work':[0]},
                                                    index=self.id)
        self.destination_act_type_ohe.loc[:,'ohe_destin_' + self.destination_act_type] = self.destination_act_type_ohe.loc[:,'ohe_destin_' + self.destination_act_type] + 1
    
    def update_mode_counts(self):
        self.mode_counts.loc[:,'count_'+self.cur_mode] = self.mode_counts.loc[:,'count_'+self.cur_mode] + 1

    def update_origin(self):
        self.origin_act_type_ohe = pd.DataFrame(data={'ohe_origin_leisure':[0],
                                                      'ohe_origin_other':[0],
                                                      'ohe_origin_school':[0],
                                                      'ohe_origin_shop':[0],
                                                      'ohe_origin_sleep':[0],
                                                      'ohe_origin_work':[0]},
                                               index=self.id)
        self.origin_act_type_ohe.loc[:,'ohe_origin_' + self.destination_act_type] = 1
        self.origin_act_type = self.destination_act_type
def ddas_framework(X_test, ATM_model, MCM_model):
    # We start by initiating an empty df for the results
    
    results = pd.DataFrame(columns=['person_id',
                                    'destination',
                                    'mode'])

    results = results.set_index('person_id')

    for i in range (0, len(X_test)):
    
        # For each row of the X_test dataframe we define a
        # new Person object
        person = Person(X_test[i:i+1].copy())
    
        # Here we set a hard-coded limit so no agent can have
        # more than 12 trips in their activity diary
        while(person.destination_act_type != 'none'):
            if person.total_trips == 12:
                person.destination_act_type = 'none'

            # If the person has not conducted 12 trips yet, let's
            # predict the next trip
            else:
                ATM_input = pd.concat([person.soc_features,
                                       person.act_type_counts,
                                       person.origin_act_type_ohe],
                                      axis = 1)

                person.destination_act_type = ATM_model.predict(ATM_input)[0]

            person.update_destination_act_type_ohe()

            MCM_input = pd.concat([ATM_input,
                                   person.destination_act_type_ohe,
                                   person.mode_counts],
                                  axis = 1)

            person.cur_mode = MCM_model.predict(MCM_input)[0]

            result_row = pd.DataFrame(data={'destination':[person.destination_act_type],
                                            'mode':[person.cur_mode]},
                                     index=person.id)

            results = results.append(result_row)

            person.update_act_counts()
            person.update_mode_counts()
            person.update_origin()
            person.total_trips = person.total_trips + 1

    return results
init_time = time.time()

results_dt = ddas_framework(X_test, ATM_dt_model, MCM_dt_model)

end_time = time.time()
ellapsed_time = end_time - init_time

print('The DDAS framework took ' + str(ellapsed_time) + ' seconds to run')
observed_list = round(results_dt.destination.value_counts()*100/len(results_dt), 6)
observed_list
expected_list = round(test_df[['activity_destination','mode_type']].activity_destination.value_counts()*100/len(test_df),6)
expected_list
def plot_expected_and_observed(input_expected, input_observed, x_label, y_label, legend, color, dist=0):
    '''Returns plots of the given lists
    
    Keyword arguments:
    input_expected -- list of the expected values of each category
    input_observed -- list of the observed values of each category
    '''
    
    fig, axs = plt.subplots(figsize=(10, 5))
        
    chart_title = ''
    df = pd.concat([input_expected, input_observed], axis=1)
        
    N = len(df)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25

    x = df.index

    y_expected = df.iloc[:,0]
    rects_expected = axs.bar(ind, y_expected, width, color='lightgreen')

    y_observed = df.iloc[:,1]
    rects_observed = axs.bar(ind+width, y_observed, width, color=color)

    axs.set_title(chart_title)
    axs.set_ylabel(y_label, fontsize=12)
    axs.set_xlabel(x_label, fontsize=12)
    axs.set_xticks(ind+width)
    axs.set_xticklabels(x)
    axs.legend((rects_expected[0], rects_observed[0]), ('expected', legend), loc=1)
    
    def autolabel(rects, offset=0):
        for rect in rects:
            try:
                h = rect.get_height()
                if (h >= 0.1):
                    axs.text(rect.get_x()+rect.get_width()/2 + offset, 1.00*h, '%g'%round(h,1),
                             ha='center', va='bottom')
                else:
                    axs.text(rect.get_x()+rect.get_width()/2 + offset, 1.00*h, '%g'%round(h,3),
                             ha='center', va='bottom')
            except:
                pass

    autolabel(rects_expected, -dist)
    autolabel(rects_observed, +dist)
        
plot_expected_and_observed(expected_list, observed_list, 'activity types', 'proportion (%)', 'Model 1', 'lightblue', dist=0.05)
def get_actv_chains(input_df):
    '''Returns a list with the activity chains performed by each person
    (e.g. H-W-N refers to chain home-work-none)
    
    Keyword arguments:
    input_df -- dataframe with trip information
    '''
    df = input_df.reset_index()
    
    chains_list = dict.fromkeys(df.iloc[:,0].unique())
    df_list = df.to_numpy()
    
    try:
        # Read each row of the input dataframe
        for i in range(0,len(df_list)):
            # If the person id is different from the previous
            if (df_list[i,0] != df_list[i-1,0]):
                #The chain of activitis for the person is initiated at home
                chains_list[df_list[i,0]] = 'home-'+df_list[i,1]
            # If is the same person of the previous row computed
            else:
                # The chain is updated with a new activity
                chains_list[df_list[i,0]] = chains_list[df_list[i,0]]+'-'+df_list[i,1]
    # Exception to deal with the first row
    except:
        chains_list[df_list[i,0]] = 'home-'+df_list[i,1]
    
    return chains_list
chains_test = pd.Series(get_actv_chains(test_df[['activity_destination','mode_type']]))
chains_test.value_counts().head(30)
len(chains_test.unique())
chains_dt = pd.Series(get_actv_chains(results_dt))
chains_dt.value_counts()
len(chains_dt.unique())
def get_actv_chain_legnths(input_df):
    '''Returns a list the length of chain of activities
    performed by each person
    (e.g.: home-work-home is a chain of size = 2
    because there are two trips: home-work and
    work-home)
    
    Keyword arguments:
    input_df -- dataframe with trip information
    '''
    df = input_df.reset_index()
    
    len_list = dict.fromkeys(df.iloc[:,0].unique())
    df_list = df.to_numpy()
    
    try:
        # Read each row of the input dataframe
        for i in range(0,len(df_list)):
            # If the person id is different from the previous
            if (df_list[i,0] != df_list[i-1,0]):
                #The length of the person's chain is initiated at one
                len_list[df_list[i,0]] = 1
                
            # If it is the same person of the previous row computed
            else:
                # The chain is updated with a new activity
                len_list[df_list[i,0]] = len_list[df_list[i,0]] + 1
    # Exception to deal with the first row
    except:
        len_list[df_list[i,0]] = 1
    
    return len_list
len_chains_test = pd.Series(get_actv_chain_legnths(test_df[['activity_destination','mode_type']]))
len_chains_test.value_counts()
len_chains_dt = pd.Series(get_actv_chain_legnths(results_dt))
len_chains_dt.value_counts()
plot_expected_and_observed(len_chains_test.value_counts(), len_chains_dt.value_counts(), 'chain length', 'counts', 'Model 1', 'lightblue', dist=0.075)
feature_importances = permutation_importance(ATM_dt_model,
                                             X_train_ATM,
                                             Y_train_ATM,
                                             scoring='f1_micro',
                                             n_repeats=5,
                                             random_state=123)

col1 = feature_importances.importances_mean
col2 = ATM_X_columns

df = pd.DataFrame(col1, index=col2, columns=['importance'])
df.sort_values(by='importance', ascending=False, inplace=True)
df
def actv_count_validation(input_df):
    '''Returns a dictionary with activity types as keys and
    frequency counts of those activities as values
    
    Keyword arguments:
    input_df -- dataframe containing trip information
    '''
    # Gets dataframe as input, turns index into column and renames columns
    df = input_df.copy()
    df.reset_index(inplace=True)
    df.columns = ['person_id','destination','mode']
    
    # Creates a dictionary with the keys being the number of times that the
    # activity is performed
    count_dict = {}
    activity_list = ['other', 'sleep', 'none', 'school', 'work', 'shop', 'leisure']
    for activity_type in activity_list:
        count_dict[activity_type] = {1:0,2:0,3:0,4:0,
                                     5:0,6:0,7:0,8:0,9:0,
                                     10:0,11:0,12:0}
    
    # Creates a new dataframe from the count of how many times each person performs
    # each type of activity. Renames the columns of this new dataframe
    x = pd.DataFrame(df.groupby(['person_id', 'destination'])['destination'].count())
    x.rename(columns={"destination": "counts"}, inplace=True)
    
    # From the previous dataframe, creates a new count of how many people perform
    # a certain number of times each activity. For instance, how many people perform
    # only 1 time the 'work' activity, etc
    y = x.groupby(['destination', 'counts'])['counts'].count()
    
    # Puts those results into the dictionary
    for activity in df.destination.unique():
        for key in y.loc[activity].index:
            count_dict[activity][key] = y.loc[activity][key]
        residual = len(df.person_id.unique()) - sum(count_dict[activity].values())
        count_dict[activity].update({0: residual})
    
    return count_dict
activ_counts_dt = actv_count_validation(results_dt)
activ_counts_test = actv_count_validation(test_df[['activity_destination','mode_type']])
def compute_chi_square_from_values(input_expected, input_observed):
    '''Returns a list containing partial chi-square values for each
    parameter being computed (activities or models) and a total
    chi-square value being the sum of those partials.
    
    Keywork arguments:
    input_expected -- dict with expected counts for the parameter
    input_observed -- dict with observed counts for the parameter
    '''
    total_chi_square = 0
    chi_square_list = []
    
    for key in input_expected:
       
        expected = pd.Series(input_expected[key], name='expected')
        expected.sort_index(inplace=True)
        
        observed = pd.Series(input_observed[key], name='observed')
        observed.sort_index(inplace=True)
        
        chi_square = ((expected-observed-1).pow(2))/expected
        chi_square[np.isnan(chi_square)] = 0
        
        total_chi_square = total_chi_square + sum(chi_square.replace(np.inf,0))
        
        chi_square.rename('chi_square_' + key)
        chi_square_list.append([key, expected, observed, sum(chi_square.replace(np.inf,0))])
    
    return [total_chi_square, chi_square_list]
chi_square_activity_counts_dt = compute_chi_square_from_values(activ_counts_test, activ_counts_dt)[0]
print('The total chi-square value for activity counts validation in the decision tree model is '
      + str(chi_square_activity_counts_dt))
chi_square_activ_subtotals = compute_chi_square_from_values(activ_counts_test, activ_counts_dt)[1]
chi_square_activ_subtotals[0]
def mode_count_validation(input_df):
    '''Returns a dictionary with transportation modes as keys and
    frequency counts of those modes as values
    
    Keyword arguments:
    input_df -- dataframe containing trip information
    '''
    df = input_df.copy()
    df.reset_index(inplace=True)
    df.columns = ['person_id','destination','mode']
    
    count_dict = {}
    activity_list = ['other', 'sleep', 'none', 'school', 'work', 'shop', 'leisure']

    for activity_type in activity_list:
        count_dict[activity_type] = {'bike':0,
                                     'car':0,
                                     'other_mode':0,
                                     'pt':0,
                                     'walk':0}
    
    x = pd.DataFrame(df.groupby(['destination', 'mode'])['mode'].count())
    
    for activity in df.destination.unique():
        for transp_mode in x.loc[activity].index:
            count_dict[activity][transp_mode] = x.loc[activity].loc[transp_mode][0]
    
    return count_dict

mode_counts_dt = mode_count_validation(results_dt)
mode_counts_dt = mode_count_validation(results_dt)
mode_counts_test = mode_count_validation(test_df[['activity_destination','mode_type']])
def compute_chi_square_from_proportions(input_expected, input_observed):
    '''Returns a list containing partial chi-square values for each
    parameter being computed (activities or models) and a total
    chi-square value being the sum of those partials.
    
    Keywork arguments:
    input_expected -- dict with expected counts for the parameter
    input_observed -- dict with observed counts for the parameter
    '''
    total_chi_square = 0
    chi_square_list = []
    
    for key in input_expected:
        
        total_expected = sum(input_expected[key].values())
        expected_proportions = {k:v/total_expected for (k,v) in input_expected[key].items()} 
        
        total_observed = sum(input_observed[key].values())
        expected_values = {k:round(v*total_observed, 0) for (k,v) in expected_proportions.items()}
        
        expected = pd.Series(expected_values, name='expected')
        expected.sort_index(inplace=True)
        
        observed = pd.Series(input_observed[key], name='observed')
        observed.sort_index(inplace=True)
        
        chi_square = ((expected-observed-1).pow(2))/expected
        chi_square[np.isnan(chi_square)] = 0
        
        total_chi_square = total_chi_square + sum(chi_square.replace(np.inf,0))
        
        chi_square.rename('chi_square_' + key)
        chi_square_list.append([key, expected, observed, sum(chi_square.replace(np.inf,0))])
    
    return [total_chi_square, chi_square_list]
chi_square_mode_counts = compute_chi_square_from_proportions(mode_counts_test, mode_counts_dt)[0]

print('The total chi-square value for mode counts validation in this set of results is '
      + str(chi_square_mode_counts))
chi_square_mode_subtotals = compute_chi_square_from_proportions(mode_counts_test, mode_counts_dt)[1]
chi_square_mode_subtotals[0]
def plot_chi_square_results(input_list, label, color):
    '''Returns plots of the given lists
    
    Keyword arguments:
    input_list -- list to be plotted
    '''
    
    fig, axs = plt.subplots(4,2,figsize=(20, 30))
    
    for i in range(0,len(input_list)):
        
        line = (i//2)
        col = (i%2)-1
        
        chart_title = input_list[i][0]
        chi_2_value = int(input_list[i][3])
        df = pd.concat([input_list[i][1],input_list[i][2]], axis=1)
        df.drop(df[df.expected == 0].index, inplace=True)
        
        N = len(df)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.25

        x = df.index

        y_expected = df['expected']
        rects_expected = axs[line,col].bar(ind, y_expected, width, color='lightgreen')

        y_observed = df['observed']
        rects_observed = axs[line,col].bar(ind+width, y_observed, width, color=color)

        axs[line,col].set_title(chart_title+', chi square='+str(chi_2_value), fontsize=20)
        axs[line,col].set_ylabel(chart_title+' frequency', fontsize=16)
        axs[line,col].set_xlabel('counts', fontsize=16)
        axs[line,col].set_xticks(ind+width)
        axs[line,col].set_xticklabels(x, fontsize=14)
        axs[line,col].legend((rects_expected, rects_observed), ('expected', label), fontsize=16, loc=1)

        def autolabel(rects):
            for rect in rects:
                h = rect.get_height()
                if (h>0):
                    axs[line,col].text(rect.get_x()+rect.get_width()/2., 1*h, '%d'%int(h),
                                       ha='center', va='bottom', fontsize=14)
                else:
                    pass

        autolabel(rects_expected)
        autolabel(rects_observed)
plot_chi_square_results(chi_square_activ_subtotals, 'Model 1', 'lightblue')
plot_chi_square_results(chi_square_mode_subtotals, 'Model 1', 'lightblue')
end_notebook = time.time()
total_duration = end_notebook - init_notebook
print('The total duration of this notebook is '+ str(total_duration))
init_time = time.time()
ATM_dt_cv_scores_mean, ATM_dt_cv_scores_std, ATM_dt_accuracy_scores = run_cross_validation_on_trees(X_train_ATM,
                                                                                                    Y_train_ATM,
                                                                                                    depths_list,
                                                                                                    scoring='balanced_accuracy')
end_time = time.time()
ellapsed_time = end_time - init_time

print('Cross-validation took ' + str(ellapsed_time) + ' seconds to run')

# plotting accuracy
plot_cross_validation_on_trees(depths_list, 
                               ATM_dt_cv_scores_mean, 
                               ATM_dt_cv_scores_std, 
                               ATM_dt_accuracy_scores, 
                               'Balanced accuracy score per decision tree depth on ATM cross-validation data',
                               'balanced accuracy')
max_balanced_accuracy = np.amax(np.around(ATM_dt_cv_scores_mean,3))

optimal_depth_ATM = np.argmax(np.around(ATM_dt_cv_scores_mean,3)) + 1

print('The maximum balanced_accuracy score for the ATM model is '
      + str(max_balanced_accuracy)
      + ' for tree depth '
      + str(optimal_depth_ATM))
X_train_cv_ATM, X_val_ATM, Y_train_cv_ATM, Y_val_ATM = train_test_split(X_train_ATM,
                                                                        Y_train_ATM,
                                                                        test_size = 0.20,
                                                                        random_state=123)

ATM_dt_cv_model = DecisionTreeClassifier(max_depth=optimal_depth_ATM, random_state=123)
ATM_dt_cv_model.fit(X_train_cv_ATM, Y_train_cv_ATM)

Y_pred_val_ATM = ATM_dt_cv_model.predict(X_val_ATM)

class_names = ['leisure', 'none', 'other', 'school', 'shop', 'sleep', 'work']

disp = plot_confusion_matrix(ATM_dt_cv_model, X_val_ATM, Y_val_ATM,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
disp.ax_.set_title('ATM confusion matrix')
plt.show()

print(classification_report(Y_val_ATM, Y_pred_val_ATM, digits=3, labels=class_names))
init_time = time.time()
MCM_dt_cv_scores_mean, MCM_dt_cv_scores_std, MCM_dt_accuracy_scores = run_cross_validation_on_trees(X_train_MCM,
                                                                                                    Y_train_MCM,
                                                                                                    depths_list,
                                                                                                    scoring='balanced_accuracy')
end_time = time.time()
ellapsed_time = end_time - init_time

print('Cross-validation took ' + str(ellapsed_time) + ' seconds to run')

# plotting accuracy
plot_cross_validation_on_trees(depths_list, 
                               MCM_dt_cv_scores_mean, 
                               MCM_dt_cv_scores_std, 
                               MCM_dt_accuracy_scores, 
                               'Balanced accuracy per decision tree depth on MCM cross-validation data',
                               'balanced accuracy')
max_balanced_accuracy = np.amax(np.around(MCM_dt_cv_scores_mean,3))

optimal_depth_MCM = np.argmax(np.around(ATM_dt_cv_scores_mean,3)) + 1

print('The maximum balanced accuracy score for the MCM model is '
      + str(max_balanced_accuracy)
      + ' for tree depth '
      + str(optimal_depth_MCM))
X_train_cv_ATM, X_val_ATM, Y_train_cv_ATM, Y_val_ATM = train_test_split(X_train_ATM,
                                                                        Y_train_ATM,
                                                                        test_size = 0.20,
                                                                        random_state=123)
sm = SMOTE(random_state=123)
X_train_cv_smote_ATM, Y_train_cv_smote_ATM = sm.fit_resample(X_train_cv_ATM, Y_train_cv_ATM)
init_time = time.time()
ATM_dt_cv_scores_mean, ATM_dt_cv_scores_std, ATM_dt_accuracy_scores = run_cross_validation_on_trees(X_train_cv_smote_ATM,
                                                                                                    Y_train_cv_smote_ATM,
                                                                                                    depths_list,
                                                                                                    scoring='balanced_accuracy')
end_time = time.time()
ellapsed_time = end_time - init_time

print('Cross-validation took ' + str(ellapsed_time) + ' seconds to run')

# plotting accuracy
plot_cross_validation_on_trees(depths_list, 
                               ATM_dt_cv_scores_mean, 
                               ATM_dt_cv_scores_std, 
                               ATM_dt_accuracy_scores, 
                               'Balanced accuracy per decision tree depth on ATM cross-validation data',
                               'balanced-accuracy')
max_balanced_accuracy = np.amax(np.around(ATM_dt_cv_scores_mean,3))

optimal_depth_ATM = np.argmax(np.around(ATM_dt_cv_scores_mean,3)) + 1

print('The maximum balanced_accuracy score for the ATM model is '
      + str(max_balanced_accuracy)
      + ' for tree depth '
      + str(optimal_depth_ATM))
ATM_dt_cv_model = DecisionTreeClassifier(max_depth=optimal_depth_ATM, random_state=123)
ATM_dt_cv_model.fit(X_train_cv_smote_ATM,
                    Y_train_cv_smote_ATM)

Y_pred_val_ATM = ATM_dt_cv_model.predict(X_val_ATM)

class_names = ['leisure', 'none', 'other', 'school', 'shop', 'sleep', 'work']

disp = plot_confusion_matrix(ATM_dt_cv_model, X_val_ATM, Y_val_ATM,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
disp.ax_.set_title('ATM confusion matrix')
plt.show()

print(classification_report(Y_val_ATM, Y_pred_val_ATM, digits=3, labels=class_names))
X_train_cv_MCM, X_val_MCM, Y_train_cv_MCM, Y_val_MCM = train_test_split(X_train_MCM,
                                                                        Y_train_MCM,
                                                                        test_size = 0.20,
                                                                        random_state=123)
X_train_cv_smote_MCM, Y_train_cv_smote_MCM = sm.fit_resample(X_train_cv_MCM, Y_train_cv_MCM)
init_time = time.time()
MCM_dt_cv_scores_mean, MCM_dt_cv_scores_std, MCM_dt_accuracy_scores = run_cross_validation_on_trees(X_train_cv_smote_MCM,
                                                                                                    Y_train_cv_smote_MCM,
                                                                                                    depths_list,
                                                                                                    scoring='balanced_accuracy')
end_time = time.time()
ellapsed_time = end_time - init_time

print('Cross-validation took ' + str(ellapsed_time) + ' seconds to run')

# plotting accuracy
plot_cross_validation_on_trees(depths_list, 
                               MCM_dt_cv_scores_mean, 
                               MCM_dt_cv_scores_std, 
                               MCM_dt_accuracy_scores, 
                               'Balanced accuracy per decision tree depth on MCM cross-validation data',
                               'balanced accuracy')
max_balanced_accuracy = np.amax(np.around(MCM_dt_cv_scores_mean,3))

optimal_depth_MCM = np.argmax(np.around(MCM_dt_cv_scores_mean,3)) + 1

print('The maximum balanced accuracy score for the MCM model is '
      + str(max_balanced_accuracy)
      + ' for tree depth '
      + str(optimal_depth_MCM))
MCM_dt_cv_model = DecisionTreeClassifier(max_depth=optimal_depth_MCM, random_state=123)
MCM_dt_cv_model.fit(X_train_cv_smote_MCM,
                    Y_train_cv_smote_MCM)

Y_pred_val_MCM = MCM_dt_cv_model.predict(X_val_MCM)

class_names = ['bike', 'car','other_mode', 'pt', 'walk']

disp = plot_confusion_matrix(MCM_dt_cv_model, X_val_MCM, Y_val_MCM,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
disp.ax_.set_title('MCM confusion matrix, without normalization')
plt.show()

print(classification_report(Y_val_MCM, Y_pred_val_MCM, digits=3, labels=class_names))
init_time = time.time()

X_train_ATM_smote, Y_train_ATM_smote = sm.fit_resample(X_train_ATM, Y_train_ATM)
ATM_dt_model_smote = DecisionTreeClassifier(max_depth=optimal_depth_ATM, random_state=123)
ATM_dt_model_smote.fit(X_train_ATM_smote, Y_train_ATM_smote)

X_train_MCM_smote, Y_train_MCM_smote = sm.fit_resample(X_train_MCM, Y_train_MCM)
MCM_dt_model_smote = DecisionTreeClassifier(max_depth=optimal_depth_MCM, random_state=123)
MCM_dt_model_smote.fit(X_train_MCM_smote, Y_train_MCM_smote)

end_time = time.time()
ellapsed_time = end_time - init_time

print('Model training for both the ATM and MCM took ' + str(ellapsed_time) + ' seconds to run')
init_time = time.time()

results_smote = ddas_framework(X_test,ATM_dt_model_smote, MCM_dt_model_smote)

end_time = time.time()
ellapsed_time = end_time - init_time

print('The DDAS framework with SMOTE took ' + str(ellapsed_time) + ' seconds to run')
observed_list_smote = round(results_smote.destination.value_counts()*100/len(results_smote), 6)
observed_list_smote
def plot_expected_and_observed3(input_expected, input_observed1, input_observed2, x_label, y_label, color1, color2, legend1, legend2):
    '''Returns plots of the given lists
    
    Keyword arguments:
    input_expected -- list of the expected values of each category
    input_observed1 -- first list of the observed values of each category
    input_observed2 -- second list of the observed values of each category
    '''
    
    fig, axs = plt.subplots(figsize=(10, 5))
        
    chart_title = ''
    df = pd.concat([input_expected, input_observed1], axis=1)
    df = pd.concat([df, input_observed2], axis=1)
        
    N = len(df)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25

    x = df.index

    y_expected = df.iloc[:,0]
    rects_expected = axs.bar(ind-width, y_expected, width, color='lightgreen')

    y_observed1 = df.iloc[:,1]
    rects_observed1 = axs.bar(ind, y_observed1, width, color=color1)
    
    y_observed2 = df.iloc[:,2]
    rects_observed2 = axs.bar(ind+width, y_observed2, width, color=color2)

    axs.set_title(chart_title)
    axs.set_ylabel(y_label, fontsize=12)
    axs.set_xlabel(x_label, fontsize=12)
    axs.set_xticks(ind)
    axs.set_xticklabels(x)
    axs.legend((rects_expected[0], rects_observed1[0], rects_observed2[0]), ('expected', legend1, legend2), loc=1)
    
    def autolabel(rects, offset=0):
        for rect in rects:
            try:
                h = rect.get_height()
                if (h >= 0.1):
                    axs.text(rect.get_x() + (rect.get_width()/2) + offset, 1.00*h, '%g'%round(h,1),
                             ha='center', va='bottom')
                else:
                    axs.text(rect.get_x() + (rect.get_width()/2) + offset, 1.00*h, '%g'%round(h,3),
                             ha='center', va='bottom')
            except:
                pass

    autolabel(rects_expected, -0.05)
    autolabel(rects_observed1)
    autolabel(rects_observed2, 0.05)
        
plot_expected_and_observed3(expected_list, observed_list, observed_list_smote, 'activity types', 'proportion (%)', 'lightblue', 'lightsalmon',
                            'Model 1', 'Model 2')
chains_smote = pd.Series(get_actv_chains(results_smote))
chains_smote.value_counts().head(20)
len(chains_smote.unique())
len_chains_smote = pd.Series(get_actv_chain_legnths(results_smote))
len_chains_smote.value_counts()
plot_expected_and_observed(len_chains_test.value_counts(),
                           len_chains_smote.value_counts(),
                           'chain length',
                           'counts',
                           'Model 2',
                           'lightsalmon',
                            dist=0.15)
feature_importances = permutation_importance(ATM_dt_model_smote,
                                             X_train_ATM_smote,
                                             Y_train_ATM_smote,
                                             scoring='balanced_accuracy',
                                             n_repeats=5,
                                             random_state=123)

col1 = feature_importances.importances_mean
col2 = ATM_X_columns

df = pd.DataFrame(col1, index=col2, columns=['importance'])
df.sort_values(by='importance', ascending=False, inplace=True)
df
activ_counts_smote = actv_count_validation(results_smote)
chi_square_activity_counts_smote = compute_chi_square_from_values(activ_counts_test, activ_counts_smote)[0]
print('The total chi-square value for activity counts validation in the decision tree model is '
      + str(chi_square_activity_counts_smote))
chi_square_activ_subtotals = compute_chi_square_from_values(activ_counts_test, activ_counts_smote)[1]
chi_square_activ_subtotals[0]
plot_chi_square_results(chi_square_activ_subtotals, 'Model 2', 'lightsalmon')
mode_counts_smote = mode_count_validation(results_smote)
chi_square_mode_counts = compute_chi_square_from_proportions(mode_counts_test, mode_counts_smote)[0]

print('The total chi-square value for mode counts validation in this set of results is '
      + str(chi_square_mode_counts))
chi_square_mode_subtotals = compute_chi_square_from_proportions(mode_counts_test, mode_counts_smote)[1]
chi_square_mode_subtotals[0]
plot_chi_square_results(chi_square_mode_subtotals, 'Model 2', 'lightsalmon')
X_train_cv_ATM, X_val_ATM, Y_train_cv_ATM, Y_val_ATM = train_test_split(X_train_ATM,
                                                                        Y_train_ATM,
                                                                        test_size = 0.20,
                                                                        random_state=123)

sm = SMOTE(random_state=123)
X_train_cv_smote_ATM, Y_train_cv_smote_ATM = sm.fit_resample(X_train_cv_ATM, Y_train_cv_ATM)

X_train_cv_MCM, X_val_MCM, Y_train_cv_MCM, Y_val_MCM = train_test_split(X_train_MCM,
                                                                        Y_train_MCM,
                                                                        test_size = 0.20,
                                                                        random_state=123)

X_train_cv_smote_MCM, Y_train_cv_smote_MCM = sm.fit_resample(X_train_cv_MCM, Y_train_cv_MCM)
def run_cross_validation_on_forests(X, y, tree_depths, cv=5, scoring='balanced_accuracy'):
    
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    
    for depth in tree_depths:
        tree_model = RandomForestClassifier(max_depth=depth, random_state=123)
        cv_scores = cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, y).score(X, y))
        
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    
    return cv_scores_mean, cv_scores_std, accuracy_scores
depths_list = [5,10,15,20,25,30,35,40]

init_time = time.time()

ATM_dt_cv_scores_mean, ATM_dt_cv_scores_std, ATM_dt_accuracy_scores = run_cross_validation_on_forests(X_train_cv_smote_ATM,
                                                                                                      Y_train_cv_smote_ATM,
                                                                                                      depths_list,
                                                                                                      cv=2,
                                                                                                      scoring='balanced_accuracy')
end_time = time.time()
ellapsed_time = end_time - init_time

print('Cross-validation took ' + str(ellapsed_time) + ' seconds to run')
# plotting accuracy
plot_cross_validation_on_trees(depths_list, 
                               ATM_dt_cv_scores_mean, 
                               ATM_dt_cv_scores_std, 
                               ATM_dt_accuracy_scores, 
                               'Balanced accuracy per decision tree depth on ATM cross-validation data',
                               'balanced-accuracy')
max_balanced_accuracy = np.amax(np.around(ATM_dt_cv_scores_mean,3))

optimal_depth_ATM = depths_list[np.argmax(np.around(ATM_dt_cv_scores_mean,3))]

print('The maximum balanced_accuracy score for the ATM model is '
      + str(max_balanced_accuracy)
      + ' for tree depth '
      + str(optimal_depth_ATM))
ATM_rf_cv_model = RandomForestClassifier(max_depth=optimal_depth_ATM, random_state=123)
ATM_rf_cv_model.fit(X_train_cv_smote_ATM,
                    Y_train_cv_smote_ATM)

Y_pred_val_ATM = ATM_rf_cv_model.predict(X_val_ATM)

class_names = ['leisure', 'none', 'other', 'school', 'shop', 'sleep', 'work']

disp = plot_confusion_matrix(ATM_rf_cv_model, X_val_ATM, Y_val_ATM,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
disp.ax_.set_title('ATM confusion matrix')
plt.show()

print(classification_report(Y_val_ATM, Y_pred_val_ATM, digits=3, labels=class_names))
init_time = time.time()

MCM_dt_cv_scores_mean, MCM_dt_cv_scores_std, MCM_dt_accuracy_scores = run_cross_validation_on_forests(X_train_cv_smote_MCM,
                                                                                                      Y_train_cv_smote_MCM,
                                                                                                      depths_list,
                                                                                                      cv=2,
                                                                                                      scoring='balanced_accuracy')
end_time = time.time()
ellapsed_time = end_time - init_time

print('Cross-validation took ' + str(ellapsed_time) + ' seconds to run')
# plotting accuracy
plot_cross_validation_on_trees(depths_list, 
                               MCM_dt_cv_scores_mean, 
                               MCM_dt_cv_scores_std, 
                               MCM_dt_accuracy_scores, 
                               'Balanced accuracy per decision tree depth on ATM cross-validation data',
                               'balanced-accuracy')
max_balanced_accuracy = np.amax(np.around(MCM_dt_cv_scores_mean,3))

optimal_depth_MCM = depths_list[np.argmax(np.around(MCM_dt_cv_scores_mean,3))]

print('The maximum balanced_accuracy score for the ATM model is '
      + str(max_balanced_accuracy)
      + ' for tree depth '
      + str(optimal_depth_MCM))
MCM_rf_cv_model = RandomForestClassifier(max_depth=optimal_depth_MCM, random_state=123)
MCM_rf_cv_model.fit(X_train_cv_smote_MCM,
                    Y_train_cv_smote_MCM)

Y_pred_val_MCM = MCM_rf_cv_model.predict(X_val_MCM)

class_names = ['bike', 'car','other_mode', 'pt', 'walk']

disp = plot_confusion_matrix(MCM_rf_cv_model, X_val_MCM, Y_val_MCM,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
disp.ax_.set_title('MCM confusion matrix')
plt.show()

print(classification_report(Y_val_MCM, Y_pred_val_MCM, digits=3, labels=class_names))
init_time = time.time()

X_train_ATM_smote, Y_train_ATM_smote = sm.fit_resample(X_train_ATM, Y_train_ATM)
ATM_rf_model_smote = RandomForestClassifier(max_depth=optimal_depth_ATM, random_state=123)
ATM_rf_model_smote.fit(X_train_ATM_smote, Y_train_ATM_smote)

X_train_MCM_smote, Y_train_MCM_smote = sm.fit_resample(X_train_MCM, Y_train_MCM)
MCM_rf_model_smote = RandomForestClassifier(max_depth=optimal_depth_MCM, random_state=123)
MCM_rf_model_smote.fit(X_train_MCM_smote, Y_train_MCM_smote)

end_time = time.time()
ellapsed_time = end_time - init_time

print('Model training for both the ATM and MCM took ' + str(ellapsed_time) + ' seconds to run')
init_time = time.time()

results_rf = ddas_framework(X_test, ATM_rf_model_smote, MCM_rf_model_smote)

end_time = time.time()
ellapsed_time = end_time - init_time

print('The DDAS framework with Random Forests and SMOTE took ' + str(ellapsed_time) + ' seconds to run')
observed_list_rf = round(results_rf.destination.value_counts()*100/len(results_rf), 6)
observed_list_rf
def plot_expected_and_observed4(input_expected, input_observed1, input_observed2, input_observed3, x_label, y_label):
    '''Returns plots of the given lists
    
    Keyword arguments:
    input_expected -- list of the expected values of each category
    input_observed1 -- first list of the observed values of each category
    input_observed2 -- second list of the observed values of each category
    '''
    
    fig, axs = plt.subplots(figsize=(16, 6))
        
    chart_title = ''
    df = pd.concat([input_expected, input_observed1], axis=1)
    df = pd.concat([df, input_observed2], axis=1)
    df = pd.concat([df, input_observed3], axis=1)
        
    N = len(df)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.2

    x = df.index

    y_expected = df.iloc[:,0]
    rects_expected = axs.bar(ind-width, y_expected, width, color='lightgreen')

    y_observed1 = df.iloc[:,1]
    rects_observed1 = axs.bar(ind, y_observed1, width, color='lightblue')
    
    y_observed2 = df.iloc[:,2]
    rects_observed2 = axs.bar(ind+width, y_observed2, width, color='lightsalmon')
    
    y_observed3 = df.iloc[:,3]
    rects_observed3 = axs.bar(ind+2*width, y_observed3, width, color='violet')

    axs.set_title(chart_title)
    axs.set_ylabel(y_label, fontsize=12)
    axs.set_xlabel(x_label, fontsize=12)
    axs.set_xticks(ind+width)
    axs.set_xticklabels(x)
    axs.legend((rects_expected[0], rects_observed1[0], rects_observed2[0], rects_observed3[0]),
               ('expected', 'Model 1', 'Model 2', 'Model 3'),
               loc=1)
    
    def autolabel(rects, offset=0):
        for rect in rects:
            try:
                h = rect.get_height()
                if (h >= 0.1):
                    axs.text(rect.get_x() + (rect.get_width()/2) + offset, 1.00*h, '%g'%round(h,1),
                             ha='center', va='bottom')
                else:
                    axs.text(rect.get_x() + (rect.get_width()/2) + offset, 1.00*h, '%g'%round(h,3),
                             ha='center', va='bottom')
            except:
                pass

    autolabel(rects_expected, -0.03)
    autolabel(rects_observed1)
    autolabel(rects_observed2, 0.03)
    autolabel(rects_observed3, 0.06)
    
        
plot_expected_and_observed4(expected_list, observed_list, observed_list_smote, observed_list_rf, 'activity types', 'proportion (%)')
chains_rf = pd.Series(get_actv_chains(results_rf))
chains_rf.value_counts().head(20)
len(chains_rf.unique())
len_chains_rf = pd.Series(get_actv_chain_legnths(results_rf))
len_chains_rf.value_counts()
plot_expected_and_observed(len_chains_test.value_counts(),
                           len_chains_rf.value_counts(),
                           'chain length',
                           'counts',
                           'Model 3',
                           'violet',
                            dist=0.15)
feature_importances = permutation_importance(ATM_rf_model_smote,
                                             X_train_ATM_smote,
                                             Y_train_ATM_smote,
                                             scoring='balanced_accuracy',
                                             n_repeats=5,
                                             random_state=123)

col1 = feature_importances.importances_mean
col2 = ATM_X_columns

df = pd.DataFrame(col1, index=col2, columns=['importance'])
df.sort_values(by='importance', ascending=False, inplace=True)
df
activ_counts_rf = actv_count_validation(results_rf)
chi_square_activity_counts_rf = compute_chi_square_from_values(activ_counts_test, activ_counts_rf)[0]
print('The total chi-square value for activity counts validation in the decision tree model is '
      + str(chi_square_activity_counts_smote))
chi_square_activ_subtotals = compute_chi_square_from_values(activ_counts_test, activ_counts_rf)[1]
chi_square_activ_subtotals[0]
plot_chi_square_results(chi_square_activ_subtotals, 'Model 3', 'violet')
mode_counts_rf = mode_count_validation(results_rf)
chi_square_mode_counts = compute_chi_square_from_proportions(mode_counts_test, mode_counts_rf)[0]

print('The total chi-square value for mode counts validation in this set of results is '
      + str(chi_square_mode_counts))
chi_square_mode_subtotals = compute_chi_square_from_proportions(mode_counts_test, mode_counts_rf)[1]
chi_square_mode_subtotals[0]
plot_chi_square_results(chi_square_mode_subtotals, 'Model 3', 'violet')