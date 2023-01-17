# Import stuff. Some stuff may be unused :/

import pandas as pd

import numpy as np

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score as acc

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler as ss, RobustScaler as rs

from sklearn.model_selection import train_test_split as tts, GridSearchCV as GSCV

from sklearn.linear_model import LogisticRegression as LR

from sklearn.tree import DecisionTreeClassifier as DTC

from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC

import category_encoders as ce

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.impute import MissingIndicator, SimpleImputer
# Download data into dataframes.

train_features = pd.read_csv('../input/train_features.csv')

train_labels = pd.read_csv('../input/train_labels.csv')

test_features = pd.read_csv('../input/test_features.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')

train_features.shape, train_labels.shape, test_features.shape, sample_submission.shape
# Merge train labels for easy splitting into training and validation sets.

train = train_features.merge(train_labels)



# Split train into training & validation sets.

train, val = tts(train, train_size=0.7, test_size=0.3, 

                 stratify=train['status_group'], random_state=42)

y_train, y_val = train['status_group'], val['status_group']



# Remove labels from training and validation sets.

train = train.drop(columns='status_group')

val = val.drop(columns='status_group')



# Make sure we have the same number of columns.

train.shape, val.shape
def clean_types(data):

    """Ensure each column has the appropriate data type.

    

    Remove some unnessecary columns.

    

    Returns: DataFrame

    

    """

    

    # Make a copy to ensure our changes stick.

    df = data.copy()

    

    # Fill categorical nan values with string.

    df = df.fillna('unknown')

    

    # Convert dates to numbers.

    dates = pd.to_datetime(df['date_recorded'])

    earliest = pd.to_datetime('2000-01-01')

    years = [x.days / 365 for x in (dates - earliest)]

    df['date_recorded'] = years

    

    # Region and district codes are categorical so should be strings

    df['region_code'] = df['region_code'].astype('str')

    df['district_code'] = df['district_code'].astype('str')



    # No repeated values == no value. Repeated columns == no value. Drrrrrrop.

    df = df.drop(columns=['recorded_by', 'quantity_group'])



    # Make sure all types are appropriate.

    types = {'amount_tsh': 'float64',

             'gps_height': 'float64',

             'date_recorded': 'float64',

             'longitude': 'float64',

             'latitude': 'float64',

             'num_private': 'float64',

             'population': 'float64',

             'construction_year': 'float64',

             'public_meeting': 'str',

             'permit': 'str'

             }

    

    df = df.astype(dtype=types)



    return df
def clean_nums(data):

    """Clean numeric columns of bad nan and spurious values.

    

    Engineer age column.

    Fill missing data with local median.

    Calculate PCA for numerical columns.

    

    Returns: DataFrame

    

    """

    

    # Make a copy to ensure our changes stick.

    df = data.copy()

    

    # Get numeric column names. Latitude is special.

    num_columns = [

                    'amount_tsh',

                    'date_recorded',

                    'gps_height',

                    'longitude',

                    'num_private',

                    'population',

                    'construction_year']

    # Dict of inapropriate nan values

    nulls = {col: 0 for col in num_columns}

    nulls['latitude'] = -2.000000e-08

    num_columns += ['latitude']

    

    # Replace bad values with nan.

    for feature, null in nulls.items():

        df[feature].replace(null, np.nan, inplace=True)

        

    # Make sure 'construction_year' is reasonable. 1960 - 2020.

    filtered_years = [x if 1960 < x < 2020 else np.nan for x in df['construction_year']]

    df['construction_year'] = filtered_years

    

    # Replace nans with nearest geographical means.

    for feature in num_columns:

        mean_ = df.groupby('ward')[feature].transform('median')

        df[feature].fillna(mean_, inplace=True)

        

    for feature in num_columns:

        mean_ = df.groupby('region')[feature].transform('median')

        df[feature].fillna(mean_, inplace=True)

    

    for feature in num_columns:

        mean_ = df[feature].median()

        df[feature].fillna(mean_, inplace=True)

    

    #Create age column

    df['age'] = df['date_recorded'] - df['construction_year']

    df = df.drop(columns=['date_recorded', 'construction_year'])

    

#     pca_data = PCA(n_components=6).fit_transform(df[num_columns])



#     df = df.drop(columns=num_columns)



#     for i in range(pca_data.shape[1]):

#         df[f'pc{i}'] = pca_data[:,i]



    return df
def clean_cats(data):

    """Clean categorical data.

    

    Replace various nan values with 'other'.

    Remove features with low frequency of incidence.

    

    Returns: DataFrame

    

    """



    # Make a copy to ensure our changes stick.

    df = data.copy()

    

    cat_columns = df.select_dtypes(exclude='number').columns.tolist()

    

    # Standardize capitaliZation.

    df[cat_columns] = df[cat_columns].applymap(lambda x: x.lower())

    

    # replace various nan names with nan.

    other_nans = ['not known', 'unknown', 'none', '-', '##', 'not kno', 'unknown installer']

    df = df.replace(other_nans, np.nan)

    

    # Low frequency values -> nan.

    for feature in cat_columns:

        keepers = df[feature].value_counts()[df[feature].value_counts() > 100].index.tolist()

        copied = df[feature].copy()

        copied[~copied.isin(keepers)] = np.nan

        df[feature] = copied

    

    # All categorical nans == 'other'.

    df[cat_columns].fillna('other', inplace=True)



    return df    
# Apply all the cleaning steps to train, valitation and test data.

X_train = clean_cats(clean_nums(clean_types(train)))

X_val = clean_cats(clean_nums(clean_types(val)))

test = clean_cats(clean_nums(clean_types(test_features)))



# Make sure we still have equal columns for each set.

X_train.shape, X_val.shape, test.shape
# Make a pipline for easy iteration of hyperparameters.

# Start with a baseline model and adjust from there.

pipeline = make_pipeline(ce.OrdinalEncoder(),

                         SimpleImputer(),

                         ss(),

                         RFC(n_jobs=-1,

                             n_estimators=100,

                             verbose=1,

#                              min_samples_leaf=25,

#                              min_samples_split=25,

#                              max_features=.9,

#                              criterion='entropy',

#                              random_state=42,

#                              max_depth=75,

                             ))
# Fit and evaluate our model.

pipeline.fit(X_train, y_train)

pipeline.score(X_val, y_val)
# Save submission if the model makes good predictions.

# predicted = pipeline.predict(test)

# submission = sample_submission.copy()

# submission['status_group'] = predicted

# submission.to_csv('sub_13.csv', index=False)