CATEGORICAL_FEATURES = ['Sex', 'Pclass']

NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch']

COL_ID = 'PassengerId'

COL_LABEL = 'Survived'



RANDOM_SEED = 1234

CROSS_FOLDS = 5



PARAMETERS_RANDOMFOREST = [

    {

        'randomforest__n_estimators': [10, 100, 500, 1000],

        'randomforest__criterion': ['gini', 'entropy'],

        'randomforest__min_samples_split': [2, 4, 8, 16]

    }

]

PARAMETERS_SVM = [

    {

        'svm__kernel': ['rbf', 'linear', 'sigmoid'],

        'svm__C': [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],

        'svm__gamma': [0.001, 0.01, 0.1, 1.0]

    }

]
import copy

import numpy as np

import pandas as pd

import sklearn.model_selection

import sklearn.ensemble

import sklearn.impute

import sklearn.metrics

import sklearn.pipeline

import sklearn.compose

import sklearn.svm



np.random.seed( RANDOM_SEED )
df_raw = pd.read_csv( '../input/train.csv', index_col=COL_ID )

df_raw.head()
# get inputs

inputs = list( CATEGORICAL_FEATURES )

inputs.extend( NUMERICAL_FEATURES )



# expected types

col_types = {col: str for col in CATEGORICAL_FEATURES}

col_types.update( {col: float for col in NUMERICAL_FEATURES} )



# check for missing features in test set

missing_field_features = []

for col in inputs:

    if df_raw[col].isnull().values.any():

        missing_field_features.append( col )



# define consistent helper for train/test

def preprocess_input( df_input: pd.DataFrame ) -> pd.DataFrame:

    # get field of interest

    df_preprocessed = df_input[inputs]



    # ensure expected types

    df_preprocessed = df_preprocessed.astype( col_types )



    # flag any missing fields with a new column

    for col in missing_field_features:

        df_preprocessed['{}_missing'.format( col )] = df_preprocessed[col].isnull()

            

    return df_preprocessed



df_X = preprocess_input( df_raw )

        

# get outputs 

df_y = df_raw[COL_LABEL]



# show preview

display( pd.concat( [df_X, df_y], axis=1 ).head() )
pipeline = sklearn.pipeline.Pipeline(

    [

        (

            'preprocess',

            sklearn.compose.ColumnTransformer(

                [

                    (

                        'onehot',

                        sklearn.preprocessing.OneHotEncoder(),

                        CATEGORICAL_FEATURES + ['{}_missing'.format( col ) for col in missing_field_features]

                    ),

                    (

                        'scale',

                        sklearn.preprocessing.StandardScaler(),

                        NUMERICAL_FEATURES

                    )

                ],

                remainder='passthrough'

            )

        ),

        (

            'imputer',

            sklearn.impute.SimpleImputer()

        )

    ]

)



df_preprocess_preview = pd.DataFrame( 

    index=df_X.index,

    data=pipeline.fit_transform( df_X )

)

df_preprocess_preview.head()
pipeline_randomforest = copy.deepcopy( pipeline )

pipeline_randomforest.steps.append( ('randomforest', sklearn.ensemble.RandomForestClassifier() ) )



search_randomforest = sklearn.model_selection.GridSearchCV(

    pipeline_randomforest,

    PARAMETERS_RANDOMFOREST,

    cv=CROSS_FOLDS,

    refit=True

)



search_randomforest.fit( df_X, df_y )
model_randomforest = search_randomforest.best_estimator_

print( 'Top Random Forest model params: {}'.format( search_randomforest.best_params_ ) )

print( 'Top Random Forest model scores: {}'.format(search_randomforest.best_score_ ) )
pipeline_svm = copy.deepcopy( pipeline )

pipeline_svm.steps.append( ('svm', sklearn.svm.SVC()) )



search_svm = sklearn.model_selection.GridSearchCV(

    pipeline_svm,

    PARAMETERS_SVM,

    cv=CROSS_FOLDS,

    refit=True,

)



search_svm.fit( df_X, df_y )
model_svm = search_svm.best_estimator_

print( 'Top SVM model params: {}'.format( search_svm.best_params_ ) )

print( 'Top SVM model scores: {}'.format( search_svm.best_score_ ) )
# load

df_test = pd.read_csv( '../input/test.csv', index_col=COL_ID )



# preprocess

df_X_test = preprocess_input( df_test )



# inference

results = model_svm.predict( df_X_test )



# write out

df_results = pd.DataFrame( results, index=df_X_test.index, columns=[COL_LABEL] )

df_results.to_csv( 'test_results.csv' )

display( df_results.head() )