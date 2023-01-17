import matplotlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


df = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

pd.set_option('display.max_columns', 500)
print(df.shape) # Shape of the data
df.head(10) # See the first 3 values of the df
print('Dtypes count:' + '\n', df.dtypes.value_counts())
columns_object = df.columns[df.dtypes == object]
print('Columns wich could have a problem :', \
       columns_object) # Columns wich need treatment beacause they are object type
def correct_targets(df):
    # Making groups by household
    all_equal_groups_1 = df.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
    
    # Selection of households were targets are not equal for all members
    all_not_equal_groups_1 = all_equal_groups_1[all_equal_groups_1 != True]

    for household in all_not_equal_groups_1.index:
        # We assumed that the correct label is the label of the head of the household, this is one possible approach
        true_target = int(df[(df['idhogar'] == household) & (df['parentesco1'] == 1.0)]['Target'])
        # Setting the correct tag for every member of the household
        df.loc[df['idhogar'] == household, 'Target'] = true_target
    
    all_equal_groups_2 = df.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
    all_not_equal_groups_2 = all_equal_groups_2[all_equal_groups_2 != True]
    
    n_corrected = len(all_not_equal_groups_1) - len(all_not_equal_groups_2)
    print("Number of targets corrected :", n_corrected)
    
    return df
    
df = correct_targets(df)
df[columns_object[2]].head(5)
(df['hogar_nin'] + df['hogar_adul'] == df['hogar_total']).all()
# Testing if the 'total' is the sum of 'min' +'adult'
inf_19_sup_65 = df['hogar_nin'] + df['hogar_mayor']
sup_19_inf_65 = df['hogar_adul'] - df['hogar_mayor']
dependecy = inf_19_sup_65*1.0 / sup_19_inf_65 # Recalculates the dependecy
dependecy.head(5)
dependecy = dependecy.replace([np.inf, -np.inf], np.nan) # Replaces all inf with NaN
dependecy.head(5)
dependecy.nlargest() # Gives the heighest dependecy rate of all families without considering inf
dependecy = dependecy.fillna(dependecy.nlargest().iloc[0]) 
# Finds the largest value and fills de NaN with it
df['dependency'].head(5)
df['dependency'] = dependecy.values
df['dependency'].head(5)
def correct_dependency(df):
    inf_19_sup_65 = df['hogar_nin'] + df['hogar_mayor']
    sup_19_inf_65 = df['hogar_adul'] - df['hogar_mayor']
    dependecy = inf_19_sup_65*1.0 / sup_19_inf_65 # Recalculates the dependecy
    dependecy = dependecy.replace([np.inf, -np.inf], np.nan) # Replaces all inf with NaN
    dependecy = dependecy.fillna(dependecy.nlargest().iloc[0]) # Finds the largest value and fills de NaN with it
    df['dependency'] = dependecy.values
    return df    
df[columns_object[3]].head(5)
df[columns_object[3]] = df[columns_object[3]].replace({'no': 0, 'yes':1}).astype(float)
df[columns_object[4]].head(5)
df[columns_object[4]] = df[columns_object[4]].replace({'no': 0, 'yes':1}).astype(float)
def correct_edjefe_edjefa(df):
    df[columns_object[3]] = df[columns_object[3]].replace({'no': 0, 'yes':1}).astype(float)
    df[columns_object[4]] = df[columns_object[4]].replace({'no': 0, 'yes':1}).astype(float)
    return df
list_na = df.columns[df.isnull().any()].tolist() #It's a list of all columns that have NAN
for column in list_na:
    series = df[column]
    n_null = series.isnull().sum()
    print('The column ' + column + ' has ' + str(n_null) + ' null values.')
df[['v2a1', 'tipovivi1', 'tipovivi4', 'tipovivi5']].head(5)
count = 0
n_ret = 0
for index, row in df.iterrows():
    if np.isnan(row['v2a1']) == True and row['tipovivi1'] == 1 or row['tipovivi4'] == 1 or row['tipovivi5'] == 1:
        df.loc[index,'v2a1'] = 0
        n_ret += 1
        
print('The amount of v2a1 changed was :', n_ret)
def correct_v2a1(df):
    count = 0
    n_ret = 0
    for index, row in df.iterrows():
        if np.isnan(row['v2a1']) == True and row['tipovivi1'] == 1 or row['tipovivi4'] == 1\
        or row['tipovivi5'] == 1:
            df.loc[index,'v2a1'] = 0
            n_ret += 1
    print('The amount of v2a1 changed was :', n_ret)
    return df
df[['v18q', 'v18q1']].head(5)
count = 0
n_ret = 0
for index, row in df.iterrows():
    if row['v18q'] == 0 and np.isnan(row['v18q1']) == True:
        df.loc[index,'v18q1'] = 0
        n_ret += 1
        
print('The amount of v18q1 changed was :', n_ret)
def correct_v18q1(df):
    count = 0
    n_ret = 0
    for index, row in df.iterrows():
        if row['v18q'] == 0 and np.isnan(row['v18q1']) == True:
            df.loc[index,'v18q1'] = 0
            n_ret += 1
    print('The amount of v18q1 changed was :', n_ret)
    return df
df[['rez_esc', 'escolari']].head(5)
df['rez_esc'] = df['rez_esc'].fillna(0)
df['meaneduc'] = df['meaneduc'].fillna(0)
df['SQBmeaned'] = df['SQBmeaned'].fillna(0)
def correct_meaneduc_rez_esc(df):
    df['rez_esc'] = df['rez_esc'].fillna(0)
    df['meaneduc'] = df['meaneduc'].fillna(0)
    df['SQBmeaned'] = df['SQBmeaned'].fillna(0)
    return df
# We choose to drop all Ids
needless_col_prov = ['idhogar','Id']
df = df.drop(needless_col_prov, axis = 1)

needless_col = needless_col_prov

# We assumed that all columns with a std inferior to 0.05 should also be droped
needless_col_prov = []
for col in df.columns:
    if df[col].std() == 0: 
        needless_col_prov.append(col)
print('The following columns have zero std so they will be discarted :', needless_col_prov)
df = df.drop(needless_col_prov, axis = 1)


needless_col = needless_col + needless_col_prov
print('In total there where', len(needless_col), 'columns eleminated.')
print('The eleminated columns where the following ones :', needless_col)
print('We are now considering', len(df.columns.tolist())-1, 'features.')
X = df.drop('Target', axis = 1)
y = df[['Target']]
'''
bayes_cv_tuner = BayesSearchCV( estimator = lgb.LGBMClassifier(boosting_type='gbdt', n_jobs=-1, verbose=2),
        search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'num_leaves': (2, 500),
        'max_depth': (0, 500),
        'min_child_samples': (0, 200),
        'max_bin': (100, 100000),
        'subsample': (0.01, 1.0, 'uniform'),
        'subsample_freq': (0, 10),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'min_child_weight': (0, 10),
        'subsample_for_bin': (100000, 500000),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'scale_pos_weight': (1e-6, 500, 'log-uniform'),
        'n_estimators': (10, 10000),
        },
        scoring = 'f1_macro', cv = StratifiedKFold(n_splits=2), n_iter = 30, verbose = 1, refit = True)
'''
'''
def rfecv_opt(model, n_jobs, X, y, cv = StratifiedKFold(2)):
    rfecv = RFECV(estimator = model, step = 1, cv = cv,
                    n_jobs = n_jobs, scoring = 'f1_macro', verbose = 1)
    rfecv.fit(X.values, y.values.ravel())
    print('Optimal number of features : %d', rfecv.n_features_)
    print('Max score with current model :', round(np.max(rfecv.grid_scores_), 3))
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel('Number of features selected')
    plt.ylabel('Cross validation score (f1_macro)')
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    important_columns = []
    n = 0
    for i in rfecv.support_:
        if i == True:
            important_columns.append(X.columns[n])
        n +=1
    return important_columns, np.max(rfecv.grid_scores_), rfecv
'''
'''
def routine(X, y, n_iter_max, n_jobs):
    list_models = []
    list_scores_max = []
    list_features = []
    list_f1_score = []
    for i in range(n_iter_max):
        print('Currently on iteration', i+1, 'of', n_iter_max, '.')
        if i == 0:
            model = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                            silent = True, metric = 'None', n_jobs = n_jobs,
                            n_estimators = 8000, class_weight = 'balanced')
        else:
            print('Adjusting model.')
            X_provi = X[imp_columns]
            # Get current parameters and the best parameters    
            result = bayes_cv_tuner.fit(X_provi.values, y.values.ravel())
            best_params = pd.Series(result.best_params_)
            param_dict=pd.Series.to_dict(best_params)
            model = lgb.LGBMClassifier(colsample_bytree = param_dict['colsample_bytree'],
                          learning_rate = param_dict['learning_rate'],
                          max_bin = int(param_dict['max_bin']),
                          max_depth = int(param_dict['max_depth']),
                          min_child_samples = int(param_dict['min_child_samples']),
                          min_child_weight = param_dict['min_child_weight'],
                          n_estimators = int(param_dict['n_estimators']),
                          num_leaves = int(param_dict['num_leaves']),
                          reg_alpha = param_dict['reg_alpha'],
                          reg_lambda = param_dict['reg_lambda'],
                          scale_pos_weight = param_dict['scale_pos_weight'],
                          subsample = param_dict['subsample'],
                          subsample_for_bin = int(param_dict['subsample_for_bin']),
                          subsample_freq = int(param_dict['subsample_freq']),
                          n_jobs = n_jobs,
                          class_weight='balanced',
                          objective='multiclass'
                          )
        imp_columns, max_score, rfecv = rfecv_opt(model, n_jobs, X, y)
        list_models.append(model)
        list_scores_max.append(max_score)
        list_features.append(imp_columns)
        
    return list_models, list_scores_max, list_features
'''
'''
list_models, list_scores_max, list_features = routine(X, y, 15, 4)

index_max = list_scores_max.index(max(list_scores_max))
features = list_features[index_max]
model = list_models[index_max]
'''
model = lgb.LGBMClassifier(boosting_type='gbdt', class_weight='balanced',
        colsample_bytree=0.364429092365, learning_rate=0.11718910536,
        max_bin=75490, max_depth=312, min_child_samples=21,
        min_child_weight=7.0, min_split_gain=0.0, n_estimators=5392,
        n_jobs=15, num_leaves=249, objective='multiclass',
        random_state=None, reg_alpha=2.51960359296e-05,
        reg_lambda=10.9020792516, scale_pos_weight=0.0247756521295,
        silent=True, subsample=0.195224406679, subsample_for_bin=126252,
        subsample_freq=3)

features = ['v2a1', 'rooms', 'r4h2', 'r4h3', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3', 'tamhog', 'tamviv',\
            'escolari', 'energcocinar3', 'hogar_nin', 'hogar_adul', 'dependency', 'edjefe', 'edjefa',\
            'meaneduc', 'bedrooms', 'overcrowding', 'qmobilephone', 'lugar1', 'age', 'SQBescolari', 'SQBage',\
            'SQBedjefe', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = X_train[features]
X_test = X_test[features]
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
test_model = model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=500, verbose=200)
predictions = test_model.predict(X_test)

print('F1-macro score on train = ', f1_score(y_test, predictions, average='macro'))
print('Dtypes count:' + '\n', test_data.dtypes.value_counts())
columns_object = test_data.columns[test_data.dtypes == object]
print('Columns wich could have a problem :', \
      columns_object) # Columns wich need treatment beacause they are object type
test_data = correct_dependency(test_data) # Correcting the 'dependency' column problem
test_data = correct_edjefe_edjefa(test_data) # Correcting the 'edjefa' and 'edjefe' problem
list_na = test_data.columns[test_data.isnull().any()].tolist() 
#It's a list of all columns that have NAN
for column in list_na:
    series = test_data[column]
    n_null = series.isnull().sum()
    print('The column ' + column + ' has ' + str(n_null) + ' null values.')
test_data = correct_v2a1(test_data)
test_data = correct_v18q1(test_data)
test_data = correct_meaneduc_rez_esc(test_data)
if not test_data.columns[test_data.isnull().any()].tolist(): 
    #It's a list of all columns that have NAN
    print('There are no columns with NaN values on the testset')
X = X[features]
y = y.values.ravel()
prediction_model = model.fit(X, y)
id_column = test_data.Id

y_pred_final = prediction_model.predict(test_data[features])
file_to_submit = pd.DataFrame({'Id':id_column, 'Target':y_pred_final})
file_to_submit.to_csv('prediction.csv', index=False)
