# Import all required libraries

%matplotlib inline

import category_encoders as ce

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.float_format', '{:.2f}'.format)

pd.set_option('display.max_columns', 50)
# Import data from sources

train = pd.merge(pd.read_csv('../input/train_features.csv'),

                 pd.read_csv('../input/train_labels.csv'))

test = pd.read_csv('../input/test_features.csv')

submission = pd.read_csv('../input/sample_submission.csv')

train.shape, test.shape, submission.shape
# Split train data in train and validate

train, val = train_test_split(train, train_size=0.85, test_size=0.15, 

                              stratify=train['status_group'])
# Create function to wrangle data

def wrangle(X):

    X_Clean = X.copy() # Create copy to not pass changes to main df

    

    # Convert region_code and district_code variables to str

    X_Clean['region_code'] = X_Clean['region_code'].astype('str')

    X_Clean['district_code'] = X_Clean['district_code'].astype('str')

    

    # Convert all NaN values to unknown as they are all categorical variables

    X_Clean = X_Clean.fillna('unknown')

    

    # Drop columns not needed and are duplicates

    X_Clean = X_Clean.drop(columns=['id','recorded_by','quantity_group','payment_type'])

    

    # Convert date_recorded column to show year

    dates = pd.to_datetime(X_Clean['date_recorded'])

    X_Clean['date_recorded'] = dates.dt.year

    

    # Convert 0's to NaN in construction year, gps height, logitude and latitude

    numericals = ['gps_height','longitude','construction_year', 'latitude', 'population']

    for col in numericals:

        X_Clean[col] = X_Clean[col].replace(0, np.nan)

        

    # Convert latitude almost 0 to nan

    X_Clean['latitude'] = X_Clean['latitude'].replace(X_Clean['latitude'].max(), np.nan)

    

    # Update missing numerical values using ward as basis

    for col in numericals:

        replacements = X_Clean.groupby('ward')[col].transform('mean')

        X_Clean[col] = X_Clean[col].fillna(replacements)

    

    # Now the numericals have NaN values, I will replace with means associated with region

    for col in numericals:

        replacements = X_Clean.groupby('region')[col].transform('mean')

        X_Clean[col] = X_Clean[col].fillna(replacements)

        

    # Any leftover numerical features with NaN will be updated with mean

    #for col in numericals:

      #   replacements = X_Clean[col].mean()

      #  X_Clean[col] = X_Clean[col].fillna(replacements)

    

    for col in numericals:

        dist = X_Clean[col].value_counts(normalize=True)

        X_Clean.loc[X_Clean[col].isna(), col] = np.random.choice(dist.index,

                                                                 size=X_Clean[col].isna().sum(),

                                                                 p=dist.values)

        

    # Create new feature age, based on date recorded minus construction date

    X_Clean['age'] = X_Clean['date_recorded'] - X_Clean['construction_year']

        

    # Return cleaned df

    return X_Clean

    
# Run wrangle function on train, val, and test sets

train_c = wrangle(train)

val_c = wrangle(val)

test_c = wrangle(test)
# The status_group column is the target

target = 'status_group'



# Get a dataframe with all train columns except the target

train_features = train_c.drop(columns=[target])



# Get a list of the numeric features

numeric_features = train_features.select_dtypes(include='number').columns.tolist()



# Get a series with the cardinality of the nonnumeric features

cardinality = train_features.select_dtypes(exclude='number').nunique()



# Get a list of all categorical features with cardinality <= 50

categorical_features = cardinality[cardinality <= 250].index.tolist()



# Combine the lists 

features = numeric_features + categorical_features
# Create X matrices and Y target vectors

X_train = train_c[features]

Y_train = train_c[target]

X_val = val_c[features]

Y_val = val_c[target]

X_test = test_c[features]
# Use pipeline method to train decision tree classifier model

pipeline1 = make_pipeline(

    ce.OneHotEncoder(use_cat_names=True), 

    SimpleImputer(strategy='mean'), 

    DecisionTreeClassifier(max_depth=20, random_state=82)

)



# Fit on train, score on val, predict on test

pipeline1.fit(X_train, Y_train)

print('Validation Accuracy', pipeline1.score(X_val, Y_val))

y_pred = pipeline1.predict(X_test)
# Use pipeline method to train random forest classifier model

pipeline2 = make_pipeline(

    ce.OrdinalEncoder(),

    SimpleImputer(strategy='mean'),

    RandomForestClassifier(n_estimators=100, random_state=23, n_jobs=-1)

)



# Fit on train, score on val

pipeline2.fit(X_train, Y_train)

print('Validation Accuracy', pipeline2.score(X_val, Y_val))

y_pred2 = pipeline2.predict(X_test)
# Hyperparameter tuning using Random Search CV

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

# Apply random search to baseline model

rf = RandomForestClassifier()

random_p2 = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)    
# Apply encoding and scaling to train and validation data to train data set to run through model training

encoder = ce.OneHotEncoder(use_cat_names=True)

X_train_encoded = encoder.fit_transform(X_train)

X_val_encoded = encoder.transform(X_val)



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_encoded)

X_val_scaled = scaler.transform(X_val_encoded)



# Fit random search model

random_p2.fit(X_train_scaled, Y_train)



# Score model against validation data set

print('Validation Accuracy', random_p2.score(X_val_scaled, Y_val))
# Create submission file

sub = submission.copy()

sub['status_group'] = y_pred2

sub.to_csv('NDoshi_DS4_Sub3.csv', index = False)