# Basic incantations.  I'll add the others as they're needed.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Create DFs for all the .CSV files.

# In Kaggle kernels, input data files are available in the "../input/" directory. 

# This one has already been loaded with the data for the competition.

# Any results you write to the current directory are saved as output.



sample_submission = pd.read_csv('../input/sample_submission.csv')

test_features = pd.read_csv('../input/test_features.csv')

train_features = pd.read_csv('../input/train_features.csv')

train_labels = pd.read_csv('../input/train_labels.csv')
def cleanup1(X):

    """

    Minimal viable cleaning.

    

    This function gets the data in minimal working order for a logistic 

    regression. I fill up NANs (which appear only in the categorcial

    features), change datetime objects to numbers, drop one useless 

    feature and standardize the datatypes.

    

    Parameters

    ----------

    X : pandas.DataFrame (DF)

        Original, full-featured DF (train_features or test_features)

    

    Returns

    ----------

    X2 : pandas.DataFrame

        Cleaned DF

    """

 

    # Make a clean copy, to ensure we're not changing the original DF

    X2 = X.copy()

    

    # Looking at all the features with missing values, it looks like those

    # features are all categorical variables where 'unknown' would be a

    # category we can work with.  I'll replace the NANs accordingly.

    X2 = X2.fillna('unknown')

    

    # Regression on dates won't work.  Instead, I'll turn the 

    # date_recorded column into the number of years since 2000

    # (the earliest date in the training date is from ~2002, and the

    # latest from 2013.)

    dates = pd.to_datetime(X2.date_recorded)

    year2000 = pd.to_datetime('2000-01-01')

    years = [i.days/365 for i in (dates - year2000)]

    X2.date_recorded = years

    

    # region_code and district_code are int64, but they should really be

    # treated as categories (and there's only 20-30 classes in each).

    # I'll cast them as strings instead.

    X2.region_code = X2.region_code.astype('str')

    X2.district_code = X2.district_code.astype('str')

    

    # recorded_by has only one value everywhere, and is therefore useless

    X2 = X2.drop(columns='recorded_by')

    

    # To prevent data conversion warnings, I'll turn all the numerical

    # features (except id) into float64.

    

    # Also, some columns contained bool values and NANs.  

    # (e.g., public_meeting, permit)

    # I replaced the NANs with strings, which created a problem for later

    # operations that don't like heterogeneous datatypes within a single

    # column. I'll prevent this problem by casting those two features as str.

    

    type_dict = {'amount_tsh':'float64',

                 'date_recorded':'float64',

                 'gps_height':'float64',

                 'longitude':'float64',

                 'latitude':'float64',

                 'num_private':'float64',

                 'population':'float64',

                 'construction_year':'float64',

                 'public_meeting':'str',

                 'permit':'str'}

    

    X2 = X2.astype(dtype = type_dict)

    

    return X2
from sklearn.impute import MissingIndicator



def cleanup2(X):

    """

    Fixes the numerical features. 

    

    

    Each feature has different specific problems, but they usually have

    garbage values (usually zero) that should really be read as NANs.

    

    I want to fix those values, but I also want to take note of the 

    datapoints where they happened.  I do this because I assume that 

    missing values tell us something about the well that our model

    might be able to pick up later.

    

    

    Parameters

    ----------

    X : pandas.DataFrame

        DF with raw numerical features

    

    Returns

    ----------

    X2 : pandas.DataFrame

         DF with cleaned numerical features and a new matrix of former

         garbage locations within those features.

    

    """

    

    

    # Make a clean copy, to ensure we're not changing the original DF

    X2 = X.copy()

    

    # I make a list of the numerical columns and a dict of their 

    # garbage values that really should be nulls

    numericals = ['amount_tsh',

                    'date_recorded',

                    'gps_height',

                    'longitude',

                    'latitude',

                    'num_private',

                    'population',

                    'construction_year']



    null_values = {'amount_tsh':0,

                     'date_recorded':0,

                     'gps_height':0,

                     'longitude':0,

                     'latitude':-2.000000e-08,

                     'num_private':0,

                     'population':0,

                     'construction_year':0}



    # I replace all garbage values with NANs.

    for feature, null in null_values.items():

        X2[feature] = X2[feature].replace(null, np.nan)



    # construction_year occasionally claims years far in the future, and 

    # could presumably also contain years way in the past.  I'll turn anything

    # not between 1960 and 2019 into a NAN.

    X2['construction_year'] = [i if 1960 < i < 2019 else np.nan for i in X2['construction_year']]

    

    

    # Creating indicator columns.

    # ---------------------------------------------------------------

    # These columns mark the locations of all the NANs 

    # in the numericals. Note that MissingIndicator returns a numpy array.

    

    indicator = MissingIndicator()

    trash_array = indicator.fit_transform(X2[numericals]) # Bool array

    trash_array = trash_array.astype('float64')     # Float64 array



    # Create a titles for the columns in num_trashmarker

    trashy_names = [numericals[i] + '_trash' for i in indicator.features_]



    # Create num_trashmarker

    trash_df = pd.DataFrame(trash_array, columns=trashy_names)



    # I add trash_df to X2

    X2 = pd.concat([X2,trash_df], sort=False, axis=1)

    

    

    # Fixing the numerical columns.

    # ---------------------------------------------------------------

    # Whenever possible, a good replacement value for a NAN is the 

    # mean or median value for the geographic region around it.



    # Replaces the NANs in a ward with the mean of the other rows in that 

    # same ward. If all the rows in a ward are NANs, though, they remain.

    for feature in numericals:

        replacements = X2.groupby('ward')[feature].transform('mean')

        X2[feature] = X2[feature].fillna(replacements)



    # Replaces the NANs in a region with the mean of the other rows in that 

    # same region (which are much larger than wards)

    for feature in numericals:

        replacements = X2.groupby('region')[feature].transform('mean')

        X2[feature] = X2[feature].fillna(replacements)

    

    # Replaces any remaining NANs with the median value for the whole dataset

    for feature in numericals:

        replacements = X2[feature].median() # Single number, not array

        X2[feature] = X2[feature].fillna(replacements)

    

    return X2
def cleanup3(X):

    """

    Fixes the categorical features. 

    

    

    Each feature has different specific problems, but they usually have

    garbage values (usually 'unknown') that should really be read as NANs.

    

    This function cleans up garbage, clusters together different labels

    that should be equivalent but are coded differently (e.g., different

    spellings of the same thing), and removes labels with so few members

    that they're unlikely to be informative.

    

    

    Parameters

    ----------

    X : pandas.DataFrame

        DF with raw categorical features, except for the changes

        already included in cleanup1.

    

    Returns

    ----------

    X2 : pandas.DataFrame

         DF with cleaned categorical features.

    

    """

    

    # Make a clean copy, to ensure we're not changing the original DF

    X2 = X.copy()

    

    # Create list of categorical features

    categoricals = X2.select_dtypes(exclude='number').columns.tolist()



    # Make all strings lowercase, to collapse together some of the categories

    X2[categoricals] = X2[categoricals].applymap(lambda x: x.lower())



    # Replace common NAN values

    nan_list = ['not known','unknown','none','-','##','not kno','unknown installer']

    X2 = X2.replace(nan_list, np.nan)



    # Any feature values with fewer than 100 rows gets turned into a NAN

    for feature in X2[categoricals]:

        # Determine which feature values to keep

        to_keep = X2[feature].value_counts()[X2[feature].value_counts() > 100].index.tolist()

        # Turn those into NANs (using a copy, to prevent warnings)

        feature_copy = X2[feature].copy()

        feature_copy[~feature_copy.isin(to_keep)] = np.nan

        X2[feature] = feature_copy



    # Fix all NANs

    X2[categoricals] = X2[categoricals].fillna('other')

    

    

    return X2
from sklearn.preprocessing import RobustScaler

def cleanup4(X):

    """

    Gets rid of mostly useless features, adds a couple of engineered ones,

    and standardizes the numericals. 

    

    Parameters

    ----------

    X : pandas.DataFrame

        DF cleaned with cleanup 1-3

    

    Returns

    ----------

    X2 : pandas.DataFrame

    

    """

    

    # Make a clean copy, to ensure we're not changing the original DF

    X2 = X.copy()

    

    garbage = ['longitude','latitude','construction_year_trash',

              'latitude_trash','gps_height_trash',

               'extraction_type_group','extraction_type_class',

               'region_code','waterpoint_type_group','source_type',

              'payment_type','quality_group','quantity_group']

    

    X2 = X2.drop(columns=garbage)

    

    X2['age'] = X2['date_recorded'] - X2['construction_year']



    numericals = ['amount_tsh',

                    'date_recorded',

                    'gps_height',

                    'num_private',

                    'population',

                    'construction_year',

                    'age']



    scaler = RobustScaler()

    nums_scaled = scaler.fit_transform(X2[numericals])

    nums_scaled = pd.DataFrame(nums_scaled, columns=numericals)

    X2[numericals] = nums_scaled

    

    return X2
# X_train is the matrix of features that will go into the logistic regression.

# It exists at various points as a dataframe or numpy array

X_train = cleanup1(train_features)

y_train = train_labels['status_group']



# This command produces a series of the categorical features, calculates their cardinality

# (number of unique values), sorts the features by cardinality, extracts the feature names

# (indices), turns those indexes into a list, and takes all but the 6 with highest cardinality. 

cols_to_keep = X_train.select_dtypes(exclude='number').nunique().sort_values().index.tolist()[:-6]

X_train = X_train[cols_to_keep]
# In this cell I define a pipeline that will one-hot encode X_train, then

# feed it to the logistic regression.



from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline



# The parameters of the regression were chosen by 

# trial and error with GridSearchCV in a separate notebook.

pipe = make_pipeline(

    OneHotEncoder(categories='auto'),

    LogisticRegression(solver='lbfgs', multi_class='ovr',

                      max_iter=500))
%%time

pipe.fit(X_train,y_train)
from sklearn.metrics import accuracy_score



# What's the accuracy of this prediction, measured against the training dataset?

y_pred = pipe.predict(X_train)

accuracy_score(y_train, y_pred)
from category_encoders.target_encoder import TargetEncoder

def target_encode_cats(X, X_train, cats, train_labels):

    """

    Target encodes a DF of categorical features, based on the three

    component vectors of y_true.  Target encoding is designed to work with

    binary labels; in order to make it work with a vector that has three

    values, I target encode against a binary version of each and then

    concatenate the results.



    Parameters

    ----------

    X : pandas.DataFrame

        Dataset to be fixed

        

    cats : List of categorical columns to encode



    train_labels : pandas.DataFrame

                    The vector of training labels



    Returns

    ----------

    X2 : pandas.DataFrame

            Fixed vector



    """

    # Make a clean copy, to ensure we're not changing the original DF

    X2 = X.copy()

    

    y_true = train_labels['status_group']

    y_works = [1.0 if x == 'functional' else 0.0 for x in y_true]

    y_broken = [1.0 if x == 'non functional' else 0.0 for x in y_true]

    y_repair = [1.0 if x == 'functional needs repair' else 0.0 for x in y_true]



    y_vectors = [y_works, y_broken, y_repair]

    X_TE_all = []



    # We want to create encoding based on the training features and 

    # labels, but apply this encoding to any vector (such as X_test)

    for i in [1,2,3]:

        # Make an encoder

        TE = TargetEncoder()

        

        # Fit it to the training data

        TE.fit(X=X_train[cats], y=y_vectors[i-1])



        # Transform the cat columns in X

        X_TE = TE.transform(X2[cats])

        

        # Give them custom names, so that the columns encoded against

        # each target vector have a different name

        X_TE = X_TE.rename(columns=(lambda x: x + '_TE' + str(i)))

        X_TE_all.append(X_TE)



    new_cats = pd.concat(X_TE_all, sort=False, axis=1)

    

    X2 = X2.drop(columns=cats)

    X2 = pd.concat([X2,new_cats], sort=False, axis=1)

    

    return X2
categoricals = ['funder',

                     'installer',

                     'wpt_name',

                     'basin',

                     'subvillage',

                     'region',

                     'district_code',

                     'lga',

                     'ward',

                     'public_meeting',

                     'scheme_management',

                     'scheme_name',

                     'permit',

                     'extraction_type',

                     'management',

                     'management_group',

                     'payment',

                     'water_quality',

                     'quantity',

                     'source',

                     'source_class',

                     'waterpoint_type',]
# Use all the cleanup steps

X_train_temp = cleanup4(cleanup3(cleanup2(cleanup1(train_features))))

X_train_new = target_encode_cats(X=X_train_temp, 

                                 X_train=X_train_temp, 

                                 cats=categoricals, 

                                 train_labels=train_labels)
%%time

from xgboost import XGBClassifier

modelxgb = XGBClassifier(objective = 'multi:softmax', booster = 'gbtree', nrounds = 'min.error.idx', 

                      num_class = 3, maximize = False, eval_metric = 'merror', eta = .1,

                      max_depth = 14, colsample_bytree = .4)



y_true = train_labels['status_group']

modelxgb.fit(X_train_new, y_true)
# Test on training data

from sklearn.metrics import accuracy_score

y_pred = modelxgb.predict(X_train_new)

accuracy_score(y_true, y_pred)