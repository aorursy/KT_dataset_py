import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import balanced_accuracy_score
APP_INPUT_LIB  = "../input/data-for-investing-type-prediction"

APP_INPUT_FILE = "investing_program_prediction_data.csv"



APP_TARGET_FEATURE = "InvType"

APP_IDENT_FEATURE  = "ID"



data = pd.read_csv(os.path.join(APP_INPUT_LIB,APP_INPUT_FILE))

data.insert(0,'ID', ['ID' + str(i)  for i in np.arange(len(data))])
display_settings = {'display.max_rows' : 20 , 'display.max_columns' : 50 , 'display.width' : 200}

for op,value in display_settings.items():

    pd.set_option(op,value)
# Print data dimensions

print("Data dimensions :")

print(data.shape)

# List of columns

print("List of columns :")

print(data.columns)
# Create feature groups

se_features = [f for f in data.columns if 'SE' in f]

ba_features = [f for f in data.columns if 'BA' in f]

pe_features = [f for f in data.columns if 'PE' in f]

ia_features = [f for f in data.columns if 'IA' in f]



#### List sample customer data - Age and Geographic location

print("Sample customer data :")

print(data[se_features].head())



#### List sample banking activity data

print("Sample banking activity data :")

print(data[ba_features].head())



#### List sample investing history data

print("Sample investing portfolio data :")

print(data[pe_features].head())



#### List sample investing activity data

print("Sample investing activity data :")

print(data[ia_features].head())
fig = plt.figure(figsize=(10,10))

sns.set(style="darkgrid")

ax = sns.countplot(x = APP_TARGET_FEATURE , data = data).set_title(APP_TARGET_FEATURE + " " + "Distribution")
label_encoders = dict()



for f in data.columns :

    if (f == APP_IDENT_FEATURE):

        continue

    f_type = data[f].dtype.name

    if (f_type == 'object'):

        f_enc = LabelEncoder() ; data[f] = f_enc.fit_transform(data[f]) ; label_encoders[f] = f_enc
from sklearn.model_selection import RepeatedStratifiedKFold

num_folds = 3 ; num_rep = 1 ; splits = dict()

rkfs_partitioner  = RepeatedStratifiedKFold(n_splits = num_folds , n_repeats = num_rep , random_state = 1234)

split_cnt = 0

for train_idxs , test_idxs in rkfs_partitioner.split(data , data[APP_TARGET_FEATURE]):

    # print("TRAIN:", len(train_idxs), "TEST:", len(test_idxs))

    i_rep  = split_cnt // num_folds

    i_part = split_cnt - i_rep * num_folds

    # print(cnt) ; print(i_rep) ; print(i_part)

    splits['R' + str(i_rep) + 'P' + str(i_part)] = {'Train' : train_idxs , 'Test' : test_idxs}

    split_cnt += 1
def report_feature_importance(i_split , feature_names , model):

    import numpy as np

    import pandas as pd

    

    feature_importance = model.feature_importances_

    indices = np.argsort(feature_importance)[::-1]

    i_feature_importance_df               = pd.DataFrame()

    i_feature_importance_df['Split']      = [i_split for f in range(len(feature_names))]

    i_feature_importance_df['Rank']       = [f for f in range(len(feature_names))]

    i_feature_importance_df['Feature']    = feature_names[indices]  

    i_feature_importance_df['Importance'] = feature_importance[indices]

    

    return i_feature_importance_df
performance_metrics    = dict()

feature_importance_df   = pd.DataFrame()



for i_split in splits.keys() :

    train_data = data.loc[splits[i_split]['Train']]

    X_train    = train_data[[i for i in train_data.columns if i not in [APP_IDENT_FEATURE , APP_TARGET_FEATURE]]]

    y_train    = train_data[APP_TARGET_FEATURE]

    i_model    = RandomForestClassifier(n_estimators = 100 ,  max_features = 'sqrt' , max_depth = 15 , random_state = 0) 

    i_model.fit(X_train,y_train)

    i_feature_importance_df = report_feature_importance(i_split , X_train.columns , i_model)

    feature_importance_df   = pd.concat([feature_importance_df, i_feature_importance_df])

    test_data = data.loc[splits[i_split]['Test']]

    X_test    = test_data[[i for i in train_data.columns if i not in [APP_IDENT_FEATURE , APP_TARGET_FEATURE]]]

    y_test    = test_data[APP_TARGET_FEATURE]

    y_test_pred = i_model.predict(X_test).flatten() 

    performance_metrics[i_split] = balanced_accuracy_score(y_test,y_test_pred)
print("Average Balanced accuracy is : %1.3f" % np.mean(list(performance_metrics.values())))

print("10 features with highest average importance")

print(feature_importance_df.groupby(['Feature'])['Importance'].mean().sort_values(ascending = False).head(10).reset_index())
import featuretools as ft

import featuretools.variable_types as vtypes

print(ft.__version__)



primitives = ft.list_primitives() # Or https://primitives.featurelabs.com/

print(primitives[primitives['type'] == 'transform'].head(5))

print(primitives[primitives['type'] == 'aggregation'].head(5))
def ftfe_create_train_num_op(X_train_num_op , fe_input_num_op , transform_primitives_set):

    

    import featuretools as ft

    import featuretools.variable_types as vtypes



    # Creating train entity set

    es_train = ft.EntitySet(id = 'invtype')

    # Adding dataframe for numeric operations

    es_train.entity_from_dataframe(entity_id = 'invtype_num_op', dataframe = X_train_num_op , index = 'ID')

    

    #print(es_train["invtype_num_op"].variables)

    

    feature_matrix_train, feature_matrix_names_train = ft.dfs(entityset = es_train, target_entity = 'invtype_num_op', 

                                                                agg_primitives = None ,

                                                                trans_primitives = transform_primitives_set , 

                                                                max_depth = 2, n_jobs = -1, chunk_size = 100 , max_features = 100 

                                                               )

    #print("feature_matrix_train :") ; print(feature_matrix_train)

    

    feature_data_train , feature_data_names_train = post_process('TRAIN' , feature_matrix_train)

    

        

    #print("feature_data_train after preprocess :") ; print(feature_data_train)

    feature_data_train  = feature_data_train[[x for x in feature_data_train.columns if x not in fe_input_num_op]]

    feature_data_names_train = [x for x in feature_data_train.columns if x not in fe_input_num_op]

    #print("Final feature_data_train :") ; print(feature_data_train)

    

    feature_matrix_train_enc, features_train_enc = ft.encode_features(feature_matrix_train, feature_matrix_names_train, include_unknown = False)

    

    return feature_data_train , features_train_enc , feature_data_names_train
def post_process(run_mode , feature_matrix, missing_threshold = 0.95, correlation_threshold = 0.95):

    

    print('Run mode : {}.'.format(run_mode))

    if(run_mode == 'TRAIN'):

        print('Feature matrix post processing missing_threshold : {} , correlation_threshold : {}'.format(missing_threshold,  correlation_threshold))

    print('Dimensionality before post processing : {}.'.format(feature_matrix.shape))

    

    #### Remove duplicated features

    start_features = feature_matrix.shape[1]

    feature_matrix = feature_matrix.iloc[:, ~feature_matrix.columns.duplicated()]

    n_duplicated   = start_features - feature_matrix.shape[1]

    print(f'Number of duplicated features : {n_duplicated}')

    

    #### Replace infinity values with missing values

    feature_matrix = feature_matrix.replace({np.inf: np.nan, -np.inf:np.nan}).reset_index()

    

    #### Treat features with missing values

    # Missing values statistics

    missing = pd.DataFrame(feature_matrix.isnull().sum())

    missing['fraction'] = missing[0] / feature_matrix.shape[0]

    missing.sort_values('fraction', ascending = False, inplace = True)

    # Missing above threshold

    missing_cols = list(missing[missing['fraction'] > missing_threshold].index)

    n_missing_cols = len(missing_cols)

    # Remove missing columns

    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in missing_cols]]

    print('Number of features with missing values above {} : {}'.format(missing_threshold , n_missing_cols))

    

    # Fill missing values with 0

    feature_matrix.fillna(0 , inplace = True)

    

    if(run_mode == 'TEST'):

        return feature_matrix



    #### Treat Zero variance features

    # Variance statistics

    unique_counts = pd.DataFrame(feature_matrix.nunique()).sort_values(0, ascending = True)

    zero_variance_cols = list(unique_counts[unique_counts[0] == 1].index)

    n_zero_variance_cols = len(zero_variance_cols)

    # Remove zero variance features

    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in zero_variance_cols]]

    print('Number of zero variance features : {}'.format(n_zero_variance_cols))

    

    #### Treat highly correlated features

    # Calculate Correlations

    corr_matrix = feature_matrix.corr()

    # Extract the upper triangle of the correlation matrix

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

    # print(upper)



    # Select the features with abolute correlation value above the threshold

    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

    # print(to_drop)

    n_collinear = len(to_drop)

    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in to_drop]]

    print('Highly correlated columns removed with correlation above {} : {} '.format(correlation_threshold , n_collinear))

    

    total_removed = n_duplicated, n_missing_cols + n_zero_variance_cols + n_collinear

    

    print('Total columns removed: ', total_removed)

    print('Dimensionality after post processing: {}'.format(feature_matrix.shape))

    

    feature_names = feature_matrix.columns

    

    return feature_matrix ,feature_names

    
def ftfe_create_test_num_op(X_test , features_train_enc , feature_data_names_train):

    

    import featuretools as ft

    import featuretools.variable_types as vtypes

    

    # Creating test entity set

    es_test = ft.EntitySet(id = 'invtype')

    # Adding dataframe for numeric operations 

    es_test.entity_from_dataframe(entity_id = 'invtype_num_op', dataframe = X_test , index = 'ID')

    feature_matrix_test = ft.calculate_feature_matrix(features = features_train_enc , entityset = es_test , n_jobs = 1)



    #print("feature_matrix_test :") ; print(feature_matrix_test)

    

    feature_data_test = feature_matrix_test[[x for x in feature_matrix_test.columns if x in feature_data_names_train]]

    #print(feature_data_test)

    feature_data_test = post_process('TEST' , feature_data_test)

    

    

    #print("Final feature_data_test :") ; print(feature_data_test)

    

    return feature_data_test
# Top 3 important business activity input features

fe_input_num_op          = ['BA3','BA7','BA4']

# Top 4 feaures regardless business meaning

fe_input_num_op          = ['BA3','BA7','BA4','SE1']

# Top 2 important business activity input features

fe_input_num_op          = ['BA3','BA7']



transform_primitives_set = ['subtract_numeric','divide_numeric']
performance_metrics    = dict()

feature_importance_df   = pd.DataFrame()

for i_split in splits.keys() :

    # i_split = 'R0P0'

    train_data = data.loc[splits[i_split]['Train']]

    X_train    = train_data[[i for i in train_data.columns if i not in [APP_TARGET_FEATURE]]]

    y_train    = train_data[APP_TARGET_FEATURE]

    

    # Add train num op features

    X_train_num_op    = train_data[[APP_IDENT_FEATURE] + fe_input_num_op]

    ftfe_train_num_op, ftfe_train_num_op_enc, ftfe_train_num_op_fnames  = ftfe_create_train_num_op(X_train_num_op , fe_input_num_op , transform_primitives_set)

    print("num_op :{0} features added".format(len(ftfe_train_num_op_fnames)))

    if(len(ftfe_train_num_op_fnames) > 1):

        X_train  = pd.merge(X_train ,ftfe_train_num_op , on = ['ID'] , how = 'inner' ).drop(APP_IDENT_FEATURE , axis=1)



    i_model  = RandomForestClassifier(n_estimators = 100 ,  max_features = 'sqrt' , max_depth = 15 , random_state = 0) 

    i_model.fit(X_train,y_train)

    

    i_feature_importance_df = report_feature_importance(i_split , X_train.columns , i_model)

    feature_importance_df   = pd.concat([feature_importance_df, i_feature_importance_df])

    

    test_data = data.loc[splits[i_split]['Test']]

    X_test    = test_data[[i for i in test_data.columns if i not in [APP_TARGET_FEATURE]]]

    y_test    = test_data[APP_TARGET_FEATURE]



    # Add test num op features    

    X_test_num_op = test_data[[APP_IDENT_FEATURE] + fe_input_num_op]

    ftfe_test_num_op = ftfe_create_test_num_op(X_test_num_op , ftfe_train_num_op_enc , ftfe_train_num_op_fnames)

    X_test  = pd.merge(X_test ,ftfe_test_num_op , on = ['ID'] , how = 'inner' ).drop(APP_IDENT_FEATURE , axis = 1)

    

    y_test_pred = i_model.predict(X_test).flatten() 

    performance_metrics[i_split] = balanced_accuracy_score(y_test,y_test_pred)

    

print("Average Balanced accuracy is : %1.3f" % np.mean(list(performance_metrics.values())))

print("10 features with highest average importance")

print(feature_importance_df.groupby(['Feature'])['Importance'].mean().sort_values(ascending = False).head(10).reset_index())