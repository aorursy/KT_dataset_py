__author__ = 'Sethuramalingam S'
__email__ = 'sethu.iit@gmail.com'
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
input_train_file = '/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv'
input_test_file = '/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv'
def load_dataset(train_file, test_file):
    '''
    Concats train, test data; sets 'target' to NaN for the test data;
    '''
    df_train = pd.read_csv(train_file)
    print("df_train.shape:{}".format(df_train.shape))

    df_test = pd.read_csv(test_file)
    print("df_test.shape:{}".format(df_test.shape))
    
    df_test['target'] = np.nan
    
    #Concatenating train and test
    df = pd.concat([df_train, df_test])
    print("df.shape:{}".format(df.shape))
    
    return df
    
    
def remove_columns(df, columns_to_remove):
    '''
    removes the columns_to_remove from the input dataframe
    '''
    return df.drop(columns=columns_to_remove)


def create_mean_std_features(df, columns):
    '''
    Creates new columns with the mean, std for the 'target' group by the input column
    '''
    print("df columns:{}".format(df.columns))
    
    for column in columns:
        new_mean = column + '_mean'
        new_std = column + '_std'

        #Create a new dataframe by grouping by the input column and calculating mean, std for the mean value
        df_column = df.loc[df['target'].notna()].groupby([column])['target'].agg(['mean', 'std']).rename(columns={'mean': new_mean, 'std': new_std}).fillna(0.0).reset_index()

        #Joining the above dataframe with the original dataframe 
        df = pd.merge(df, df_column, how='left', on=[column])

        df[new_mean] = df[new_mean].fillna(0.0)
        df[new_std] = df[new_std].fillna(0.0)
    
    return df


def create_one_hot_encodings(df):
    '''
    Creates one-hot encodings for the non-numerical columns in the input dataframe
    '''
    return pd.get_dummies(df, columns=[c for c in df.columns if df[c].dtype == 'object'])

def grid_search_knn_params(X_train, y_train, grid_search_params):
    '''
    Grid search for best parameters for the KNeighboursClassifier
    '''
    #for knn scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    knn_grid = GridSearchCV(KNeighborsClassifier(), grid_search_params, cv=5, scoring='neg_log_loss', n_jobs=-1, verbose=True)
    knn_grid.fit(X_train_scaled, y_train)
    return (knn_grid.best_params_, knn_grid.best_score_)

def grid_search_dt_params(X_train, y_train, grid_search_params):
    '''
    Grid search for best parameters of Decision Trees
    '''
    dt_grid = GridSearchCV(DecisionTreeClassifier(), grid_search_params, cv=5, scoring='neg_log_loss', n_jobs=-1, verbose=True)
    dt_grid.fit(X_train, y_train)
    return (dt_grid.best_params_, dt_grid.best_score_)
    
    
def create_decisiontree_model(X, y, dt_params):
    '''
    Creates a decision tree on the input X and y data with the input parameters
    '''
    model = DecisionTreeClassifier(
        criterion=dt_params['criterion'],
        splitter='best',
        max_depth=dt_params['max_depth'],
        min_samples_split=dt_params['min_samples_split'],
        min_samples_leaf=dt_params['min_samples_leaf']
    )

    return model.fit(X, y)

def create_knn_model(X,y, knn_params):
    '''
    Create a knn model on the input X and y data for the input 'k'
    '''
    #for knn scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KNeighborsClassifier(n_neighbors=knn_params['n_neighbors'], weights=knn_params['weights'],metric=knn_params['metric'], n_jobs=-1)
    #model = KNeighborsClassifier(n_neighbors=knn_params['n_neighbors'], n_jobs=-1)
    return model.fit(X_scaled, y)
    
def calculate_log_loss(model, X_test, y_test):
    return log_loss(y_test, model.predict_proba(X_test)[:, 1])
#main logic

#Reads the train and test file, concatenates them into a dataframe
df = load_dataset(input_train_file, input_test_file)
print("On loading df.shape:\n{}".format(df.head()))

#Removes columns based on feature selection
df = remove_columns(df, ['workclass','race', 'sex','native-country','education', 'fnlwgt'])
print("After removing columns:\n{}".format(df.head()))

#Add new columns for mean, std of the target variable grouping by the input list of columns
df = create_mean_std_features(df, ['education-num', 'marital-status', 'occupation', 'relationship'])

#Creates one-hot encodings for the categorical features
df = create_one_hot_encodings(df)
print("After creating one-hot encoding:{}".format(df.shape))

#Create X and y values
X = df.loc[df['target'].notna()].drop(columns=['target','uid'])
y = df.loc[df['target'].notna()]['target']
print("X shape:{}".format(X.shape))
print("y shape:{}".format(y.shape))

# #Split the data into train and hold-out/validation data
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.33, random_state=17)

#Train the decision tree model on the X_train and y_train
dt_params = { "criterion":'gini',
    "max_depth":7,
    "min_samples_split":8,
    "min_samples_leaf":28
    }
dt_model = create_decisiontree_model(X_train, y_train, dt_params)

print("Log loss:{}".format(calculate_log_loss(dt_model, X_holdout, y_holdout)))

#Train the decision tree model on the entire train data
dt_model_full = create_decisiontree_model(X, y, dt_params)

#Test data extraction
test_data = df.loc[df['target'].isna()].drop(columns=['uid','target'])
print("Test data shape:{}".format(test_data.shape))

#Prediction based on the DT model trained on full train data
p = dt_model_full.predict_proba(test_data)[:, 1]

#Create a dataframe for submission
df_submit = pd.DataFrame({
    'uid':df.loc[df['target'].isna()]['uid'],
    'target':p
})
print("Final df_subit shape:{}".format(df_submit.shape))

#Write the submission output to file
df_submit.to_csv('/kaggle/working/submit_using_MI_based_feature_selected_plus_mean_std_features_new.csv', index=False)
#main logic

#Reads the train and test file, concatenates them into a dataframe
df = load_dataset(input_train_file, input_test_file)
print("On loading df.shape:\n{}".format(df.columns))

#Removes columns based on feature selection
df = remove_columns(df, ['workclass','race', 'sex','native-country','education', 'fnlwgt'])
print("After removing columns:\n{}".format(df.columns))

#Add new columns for mean, std of the target variable grouping by the input list of columns
df = create_mean_std_features(df, ['education-num', 'marital-status', 'occupation', 'relationship'])

#Creates one-hot encodings for the categorical features
df = create_one_hot_encodings(df)
print("After creating one-hot encoding:{}".format(df.shape))

#Create X and y values
X = df.loc[df['target'].notna()].drop(columns=['target','uid'])
y = df.loc[df['target'].notna()]['target']
print("X shape:{}".format(X.shape))
print("y shape:{}".format(y.shape))

# #Split the data into train and hold-out/validation data
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.33, random_state=17)

######################## Step 0: Tune the DT params ###########################################
# dt_params = {
#     "max_depth":range(5,15),
#     "max_features" : [10,15,20,25,30,35,40,45,50],
#     "max_leaf_nodes" : [10,15,20,25,30,35,40,45,50],
#     "min_samples_leaf" : [10,15,20,25,30,35,40,45,50],
#     "min_samples_split" : [10,15,20,25,30,35,40,45,50]
# }

# (dt_best_params, dt_best_score) = grid_search_dt_params(X_train, y_train, dt_params)
# print("DT_best_params:{}".format(dt_best_params))
# print("DT_best_score:{}".format(dt_best_score))
##DT_best_params:{'max_depth': 11, 'max_features': 30, 'max_leaf_nodes': 50, 'min_samples_leaf': 40, 'min_samples_split': 30}

######################## Step 1: Train the DT model ###########################################
#Train the decision tree model on the X_train and y_train
dt_params = { "criterion":'gini',
    "max_depth":7,
    'min_samples_leaf': 28, 
    'min_samples_split': 8
    }
dt_model = create_decisiontree_model(X_train, y_train, dt_params)
print("DT Log loss:{}".format(calculate_log_loss(dt_model, X_holdout, y_holdout)))

######################## Step 2.0: Find the best parameters for the KNN model ###########################################
# knn_params = {
#     'n_neighbors': range(1, 20),
#     'weights' : ['uniform', 'distance'],
#     'metric' : ['euclidean', 'manhattan']
# }
# (knn_best_params, knn_best_score) = grid_search_knn_params(X_train, y_train, knn_params)
# print("knn_best_params:{}".format(knn_best_params))
# print("knn_best_score:{}".format(knn_best_score))
## Best params obtained: knn_best_params:{'metric': 'manhattan', 'n_neighbors': 19, 'weights': 'uniform'}

# ######################## Step 3: Train the KNN model ###########################################
knn_params = {
    'n_neighbors': 10,
    'weights': 'distance',
    'metric' : 'euclidean'
}
knn_model = create_knn_model(X_train, y_train, knn_params)
print("KNN Log loss:{}".format(calculate_log_loss(knn_model, X_holdout, y_holdout)))

# ######################## Step 3: Weighted Average ###########################################
dt_predictions = dt_model.predict_proba(X_holdout)[:,1]
knn_predictions = knn_model.predict_proba(X_holdout)[:,1]
final_predictions = dt_predictions * 0.75 + knn_predictions * 0.25

print("Final log loss:{}".format(log_loss(y_holdout, final_predictions)))

#Train the decision tree model on the entire train data
dt_model_full = create_decisiontree_model(X, y, dt_params)
knn_model_full = create_knn_model(X, y, knn_params)

#Test data extraction
test_data = df.loc[df['target'].isna()].drop(columns=['uid','target'])
print("Test data shape:{}".format(test_data.shape))

#Prediction based on the DT model and KNN models trained on full train data
dt_predictions = dt_model_full.predict_proba(test_data)[:, 1]

#Scale the test data for KNNs
scaler = StandardScaler()
test_data_scaled = scaler.fit_transform(test_data)
knn_predictions = knn_model_full.predict_proba(test_data_scaled)[:, 1]
final_predictions = dt_predictions * 0.75 + knn_predictions * 0.25

#Create a dataframe for submission
df_submit = pd.DataFrame({
    'uid':df.loc[df['target'].isna()]['uid'],
    'target':final_predictions
})
print("Final df_subit shape:{}".format(df_submit.shape))

#Write the submission output to file
df_submit.to_csv('/kaggle/working/submit_ensemble_dt0.75_knn0.25.csv', index=False)
print("knn_best_score:{}".format(knn_best_score))
