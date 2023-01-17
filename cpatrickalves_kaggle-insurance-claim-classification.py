# Loading useful Python packages for Data cleaning and Pre-processing

import numpy as np

import pandas as pd

import pandas_profiling

import matplotlib.pyplot as plt

import category_encoders as ce

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import StandardScaler

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore')

pd.set_option('display.max_columns', 150)
# loading datasets

train_df = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv')

test_df = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')
train_df.head()
data = {}

data['original'] = {'train': train_df, 'test': test_df}
train_df.info()
# There are null values?

train_df.isnull().values.any()
# Null values amount for each column

train_df.isnull().sum().sort_values(ascending=False)
null_values = train_df.isnull().sum()

null_values = round((null_values/train_df.shape[0] * 100), 2)

null_values.sort_values(ascending=False)
# Get the names of the columns that have more than 40% of null values

high_nan_rate_columns = null_values[null_values > 40].index



# Make a copy if the original datasets and remove the columns

train_df_cleaned = train_df.copy()

test_df_cleaned = test_df.copy()

train_df_cleaned.drop(high_nan_rate_columns, axis=1, inplace=True)

test_df_cleaned.drop(high_nan_rate_columns, axis=1, inplace=True)



# Remove the ID column (it is not useful for modeling)

train_df_cleaned.drop(['ID'], axis=1, inplace=True)



train_df_cleaned.info()
null_values_columns = train_df_cleaned.isnull().sum().sort_values(ascending=False)

null_values_columns = null_values_columns[null_values_columns > 0]

null_values_columns
train_df_cleaned[null_values_columns.index].info()
###### TRAIN DATASET ######



##### Numerical columns

null_values_columns_train = train_df_cleaned.isnull().sum().sort_values(ascending=False)

numerical_col_null_values = train_df_cleaned[null_values_columns_train.index].select_dtypes(include=['float64', 'int64']).columns

# for each column

for c in numerical_col_null_values:

    # Get the mean

    mean = train_df_cleaned[c].mean()

    # replace the NaN by mode

    train_df_cleaned[c].fillna(mean, inplace=True)



##### Categorical columns

categ_col_null_values = train_df_cleaned[null_values_columns_train.index].select_dtypes(include=['object']).columns

# for each column

for c in categ_col_null_values:

    # Get the most frequent value (mode)

    mode = train_df_cleaned[c].value_counts().index[0]

    # replace the NaN by mode

    train_df_cleaned[c].fillna(mode, inplace=True)

    



###### TEST DATASET ######



##### Numerical columns

null_values_columns_test = test_df_cleaned.isnull().sum().sort_values(ascending=False)

#print(null_values_columns_test)

numerical_col_null_values = list(test_df_cleaned[null_values_columns_test.index].select_dtypes(include=['float64', 'int64']).columns)

numerical_col_null_values.remove('ID')

# for each column

for c in numerical_col_null_values:

    # Get the mean

    mean = test_df_cleaned[c].mean()

    # replace the NaN by mode

    test_df_cleaned[c].fillna(mean, inplace=True)



##### Categorical columns

categ_col_null_values = test_df_cleaned[null_values_columns_test.index].select_dtypes(include=['object']).columns

# for each column

for c in categ_col_null_values:

    # Get the most frequent value (mode)

    mode = test_df_cleaned[c].value_counts().index[0]

    # replace the NaN by mode

    test_df_cleaned[c].fillna(mode, inplace=True)

    
# There are null values?

print(train_df_cleaned.isnull().values.any())

print(test_df_cleaned.isnull().values.any())
# Save the list of current columns

selected_columns = list(train_df_cleaned.columns)

selected_columns_test = selected_columns[:]

selected_columns_test.remove('target')

selected_columns_test.append('ID')



# Filter the columns in the test dataset

test_df_cleaned = test_df_cleaned[list(selected_columns_test)]



# Save the datasets in dict

data['cleaned_v1'] = {'train': train_df_cleaned.copy(), 'test':test_df_cleaned.copy()}
%%time

train_df_cleaned.profile_report(style={'full_width':True})
selected_columns = list(train_df_cleaned.columns)

# Remove the selected columns

selected_columns.remove('v12')

selected_columns.remove('v114')
# Remove the selected columns

selected_columns.remove('v125')

selected_columns.remove('v22')

selected_columns.remove('v112')

selected_columns.remove('v56')



# Save the list of current columns

selected_columns_test = selected_columns[:]

selected_columns_test.remove('target')

selected_columns_test.append('ID')



# Filter the columns in the train dataset

train_df_cleaned = train_df_cleaned[selected_columns].copy()

# Filter the columns in the test dataset

test_df_cleaned = test_df_cleaned[selected_columns_test].copy()



# Save the datasets in dict

data['cleaned_v2'] = {'train': train_df_cleaned.copy(), 'test':test_df_cleaned.copy()}
# Remove the selected columns

selected_columns.remove('v129')

selected_columns.remove('v38')

selected_columns
# Save the list of current columns

selected_columns_test = selected_columns[:]

selected_columns_test.remove('target')

selected_columns_test.append('ID')



# Filter the columns in the train dataset

train_df_cleaned = train_df_cleaned[selected_columns].copy()

# Filter the columns in the test dataset

test_df_cleaned = test_df_cleaned[selected_columns_test].copy()



# Save the datasets in dict

data['cleaned_v3'] = {'train': train_df_cleaned.copy(), 'test': test_df_cleaned.copy()}
train_df_cleaned = data['cleaned_v2']['train'].copy()

test_df_cleaned = data['cleaned_v2']['test'].copy()

train_df_cleaned.select_dtypes(include=['object']).columns
# Before encoding categorical variables we need to convert the categorical data from "object" to "category"

# Train

for col_name in train_df_cleaned.select_dtypes(include=['object']).columns:    

    train_df_cleaned[col_name] = train_df_cleaned[col_name].astype('category')



# Test

for col_name in test_df_cleaned.select_dtypes(include=['object']).columns:

    test_df_cleaned[col_name] = test_df_cleaned[col_name].astype('category')

train_df_cleaned.select_dtypes(include=['category']).describe()
##### VERSION 1

# Encoding all categorical variables with OneHot

cat_columns = ['v3', 'v24',  'v31', 'v66', 'v74', 'v75', 'v110', 'v47', 'v52','v71', 'v91', 'v107', 'v79']

ce_onehot = ce.OneHotEncoder(cols=cat_columns)



# For columns v47 and v79, the are some values only present in the train dataset. Thus, the enconding process create a different number of columns 

# in train and test dataset and prevents the model prediction. So before save the datasets I remove the extra columns 'v47_10', 'v79_18'.

# Apply the encoding

data['cleaned_transformed_CatgEncoded_v1'] = {'train':ce_onehot.fit_transform(train_df_cleaned).drop(['v47_10', 'v79_18'], axis=1), 

                                              'test': ce_onehot.fit_transform(test_df_cleaned)}
print(data['cleaned_transformed_CatgEncoded_v1']['train'].columns)

print(data['cleaned_transformed_CatgEncoded_v1']['test'].columns)
##### VERSION 2

# Encoding categorical variables with low cardinality with OneHot

low_cardinality_columns = ['v3', 'v24',  'v31', 'v66', 'v74', 'v75', 'v110']

ce_onehot = ce.OneHotEncoder(cols=low_cardinality_columns)



# Apply the encoding 

train_df_cleaned_transformed = ce_onehot.fit_transform(train_df_cleaned)

test_df_cleaned_transformed = ce_onehot.fit_transform(test_df_cleaned)
# Encoding categorical variables with high cardinality with Hashing

high_cardinality_columns = ['v47', 'v52','v71', 'v91', 'v107', 'v79']



#train_df_cleaned_transformed[high_cardinality_columns].describe().loc['unique']



ce_hash = ce.HashingEncoder(max_process=1, cols = high_cardinality_columns, n_components=12)

train_df_cleaned_transformed = ce_hash.fit_transform(train_df_cleaned_transformed)

test_df_cleaned_transformed = ce_hash.fit_transform(test_df_cleaned_transformed)
data['cleaned_transformed_CatgEncoded_v2'] = {'train': train_df_cleaned_transformed.copy(), 'test': test_df_cleaned_transformed.copy()}
train_df_cleaned_transformed.head(5)
##### VERSION 3

# Removing all categorical variables with OneHot

cat_columns = ['v3', 'v24',  'v31', 'v66', 'v74', 'v75', 'v110', 'v47', 'v52','v71', 'v91', 'v107', 'v79']



# Apply the encoding

data['cleaned_dropCatg'] = {'train':train_df_cleaned.drop(columns=cat_columns, axis=1), 

                               'test': test_df_cleaned.drop(columns=cat_columns, axis=1)}



data['cleaned_dropCatg']['train'].head()
train_df_cleaned.select_dtypes(exclude=['category']).describe()
# Plot the distribution of numerical features



# Create fig object

fig, axes = plt.subplots(2, 5, figsize=(20,8))



numerical_columns = train_df_cleaned.select_dtypes(exclude=['category']).columns

numerical_columns = list(numerical_columns)

numerical_columns.remove('target')



# Create a plot for each feature

x,y = 0,0

for i, column in enumerate(numerical_columns):

    

    sns.distplot(train_df_cleaned[column], ax=axes[x,y])

    if i < 4:

        y += 1

    elif i==4:

        x = 1

        y = 0

    else:

        y+=1
# Apply all scalings methods

scaling = {'MinMaxScaler': MinMaxScaler(),

         'RobustScaler': RobustScaler(),

         'StandardScaler': StandardScaler()

        }



# Temporarily save transformed data sets

temp_dict = {}



# Save the list of the numerical columns of the original dataset

num_cols = list(data['original']['train'].select_dtypes(exclude=['object']).columns)



# Apply all scalings in all datasets

for d in data.keys():

    print(f"Scaling dataset: {d}")

    

    # Get the list of numerical columns

    cols_train = list(data[d]['train'].select_dtypes(exclude=['category','object']).columns)

    cols_test = list(data[d]['test'].select_dtypes(exclude=['category','object']).columns)

    cols_train.remove('target')

    

    # As the encoding process of categorical features create numerical columns

    # we need to filter these columns    

    cols_train = list(set(num_cols) & set(cols_train))

    cols_test = list(set(num_cols) & set(cols_test))

    cols_test.remove('ID')

        

    # Apply Transformations

    for s in scaling.keys():

        print(f"   Applying {s}() ...")    

        

        # Make a copy of the original DFs

        train = data[d]['train'].copy()

        test = data[d]['test'].copy()

        # Apply scaling

        train[cols_train] = scaling[s].fit_transform(train[cols_train])

        test[cols_test] = scaling[s].fit_transform(test[cols_test])    

        # Save the data

        temp_dict[f"{d}_{s}"] = {'train': train.copy(), 'test': test.copy()}



# Save the new datasets in data dict        

data.update(temp_dict)        

print(data.keys())
# Importing packages

from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, log_loss

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.calibration import CalibratedClassifierCV

from xgboost.sklearn import XGBClassifier

from xgboost import plot_importance

from imblearn.over_sampling import SMOTE, ADASYN

from numpy import sort

import lightgbm as lgb

import xgboost as xgb
len(data)
# A function to run train and test for each model

def run_model(name, model, X_train, Y_train, cv_folds=5, verbose=True):   

    

    if verbose: print(f"{name}")

    

    # Use Stratified ShuffleSplit cross-validator

    # Provides train/test indices to split data in train/test sets.

    n_folds = 5

    sss = StratifiedShuffleSplit(n_splits=cv_folds, test_size=0.30, random_state=10)



    # Control the number of folds in cross-validation (5 folds)

    k=1

    

    acc = 0

    roc = 0

    log_loss_score = 0

    

    # From the generator object gets index for series to use in train and validation

    for train_index, valid_index in sss.split(X_train, Y_train):



        # Saves the split train/validation combinations for each Cross-Validation fold

        X_train_cv, X_validation_cv = X_train.loc[train_index,:], X_train.loc[valid_index,:]

        Y_train_cv, Y_validation_cv = Y_train[train_index], Y_train[valid_index]

        

        #print(f"Fold: {k}") 

        # Training the model

        try:

            model.fit(X_train_cv, Y_train_cv, eval_set=[(X_train_cv, Y_train_cv), (X_validation_cv, Y_validation_cv)], eval_metric='logloss', verbose=False )        

        except:

            try: 

                model.fit(X_train_cv, Y_train_cv, eval_set=[(X_train_cv, Y_train_cv), (X_validation_cv, Y_validation_cv)], verbose=False)        

            except:

                try:

                    model.fit(X_train_cv, Y_train_cv, verbose=False)

                except:

                    model.fit(X_train_cv, Y_train_cv)

                        

        # Get the class probabilities of the input samples        

        train_pred = model.predict(X_validation_cv)

        train_pred_prob = model.predict_proba(X_validation_cv)[:,1]

       

        acc += accuracy_score(Y_validation_cv, train_pred)

        roc += roc_auc_score(Y_validation_cv, train_pred_prob)

        log_loss_score += log_loss(Y_validation_cv, train_pred_prob)   

                

        k += 1

    

    # Compute the mean

    if verbose:

        print("Accuracy : %.4g" % (acc/(k-1)))

        print("AUC Score: %f" % (roc/(k-1)))

        print("Log Loss: %f" % (log_loss_score/(k-1)))

        print("-"*30)



    # Return the last version 

    return (model, log_loss_score/(k-1))
print(list(data.keys()))
models = {}



# From previous analysis I see that KNN, Random Forest and Extra Trees had poor results and SVM took to long to run, I'll remove them from the models list



models['LogisticRegression'] = LogisticRegression()

models['LinearDiscriminantAnalysis'] = LinearDiscriminantAnalysis()

#models['KNeighborsClassifier'] = KNeighborsClassifier(n_jobs=-1)

#models['SVM'] = SVC(probability=True)

#models['RandomForestClassifier'] = RandomForestClassifier(n_jobs=-1)

#models['ExtraTreesClassifier'] = ExtraTreesClassifier(n_jobs=-1)

models['LGBMClassifier'] = lgb.LGBMClassifier(objective='binary', 

                                              is_unbalance=True, 

                                              max_depth=30, 

                                              learning_rate=0.05, 

                                              n_estimators=500, 

                                              num_leaves=30,

                                             verbose = 0)



# The model parameters were taken from https://www.kaggle.com/rodrigolima82/kernel-xgboost-otimizado

# Thanks Rodrigo Lima for sharing his kernel

models['XGBClassifier'] = XGBClassifier(learning_rate = 0.1,

                          n_estimators = 200,

                          max_depth = 5,

                          min_child_weight = 1,

                          gamma = 0,

                          subsample = 0.8,

                          colsample_bytree = 0.8,

                          objective = 'binary:logistic',

                          n_jobs = -1,

                          scale_pos_weight = 1,

                          verbose = False,

                          seed = 32)





# When performing classification you often want to predict not only the class label, but also the associated probability. 

# This probability gives you some kind of confidence on the prediction. 

# However, not all classifiers provide well-calibrated probabilities, some being over-confident while others being under-confident. 

# Thus, a separate calibration of predicted probabilities is often desirable as a postprocessing.

#models['Calibrated_LogisticRegression'] = CalibratedClassifierCV(LogisticRegression())

#models['Calibrated_LinearDiscriminantAnalysis'] = CalibratedClassifierCV(LinearDiscriminantAnalysis())

#models['Calibrated_KNeighborsClassifier'] = CalibratedClassifierCV(KNeighborsClassifier(n_jobs=-1))

#models['Calibrated_SVM'] = CalibratedClassifierCV(models['SVM'])

# Splitting features and targets for train data

datasets = ['cleaned_transformed_CatgEncoded_v1_MinMaxScaler', 'cleaned_transformed_CatgEncoded_v1_RobustScaler', 

            'cleaned_transformed_CatgEncoded_v1_StandardScaler', 'cleaned_transformed_CatgEncoded_v2_MinMaxScaler', 

            'cleaned_transformed_CatgEncoded_v2_RobustScaler', 'cleaned_transformed_CatgEncoded_v2_StandardScaler', 

            'cleaned_dropCatg_MinMaxScaler', 'cleaned_dropCatg_RobustScaler', 'cleaned_dropCatg_StandardScaler']



results = pd.DataFrame(columns=['Dataset', 'Model', 'Logloss'])



# loop through all datasets and ML models

for d in datasets:

    train = data[d]['train']

    train_x = train.drop(['target'], axis=1)

    train_y = train['target']

    

    print(f'###### DATASET: {d} ######')

    

    for m in models.keys():

        # Train and test the model

        models[m], log_loss_result = run_model(m, models[m], train_x, train_y)  

        

        # Save Results

        results = results.append({'Dataset' : d , 'Model' : m, 'Logloss': log_loss_result} , ignore_index=True)
results.sort_values(by=['Logloss'])
#sns.stripplot(x = 'Model', y = 'Logloss', data = results, jitter = True)



plt.figure(figsize=(10,7))

chart = sns.stripplot(x = 'Model', y = 'Logloss', data = results)

chart.set_xticklabels(chart.get_xticklabels(), rotation=55)

plt.show(); 
plt.figure(figsize=(10,7))

chart = sns.stripplot(x = 'Dataset', y = 'Logloss', data = results)

chart.set_xticklabels(chart.get_xticklabels(), rotation=55)

plt.show(); 

train = data['cleaned_transformed_CatgEncoded_v1_MinMaxScaler']['train']

train_x = train.drop(['target'], axis=1)

train_y = train['target']



# train model

best_model = XGBClassifier(learning_rate = 0.1,

                    n_estimators = 200,

                    max_depth = 5,

                    min_child_weight = 1,

                    gamma = 0,

                    subsample = 0.8,

                    colsample_bytree = 0.8,

                    objective = 'binary:logistic',

                    n_jobs = -1,

                    scale_pos_weight = 1,

                    verbose = False,

                    seed = 32)

     

best_model.fit(train_x, train_y, eval_metric='logloss', verbose=False )        



fig, ax = plt.subplots(figsize=(17,15))

plot_importance(best_model, ax=ax)

plt.show()
# Fit model using each importance as a threshold

thresholds = sort(best_model.feature_importances_)



# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=7)



# Evaluate the result for several thresholds (different number of features)

for thresh in sort(list(set(thresholds))):

    # select features using threshold

    selection = SelectFromModel(best_model, threshold=thresh, prefit=True)

    select_X_train = selection.transform(X_train)

    

    # train model

    selection_model = XGBClassifier(learning_rate = 0.1,

                    n_estimators = 200,

                    max_depth = 5,

                    min_child_weight = 1,

                    gamma = 0,

                    subsample = 0.8,

                    colsample_bytree = 0.8,

                    objective = 'binary:logistic',

                    n_jobs = -1,

                    scale_pos_weight = 1,

                    verbose = False,

                    seed = 32)

     

    selection_model.fit(select_X_train, y_train, eval_metric='logloss', verbose=False )        

    

    # eval model    

    select_X_test = selection.transform(X_test)

    y_pred = selection_model.predict(select_X_test)        

    train_pred_prob = selection_model.predict_proba(select_X_test)[:,1]    

    log_loss_score = log_loss(y_test, train_pred_prob)   

    

    print("Thresh=%.3f, n=%d, logloss: %.6f" % (thresh, select_X_train.shape[1], log_loss_score))

train = data['cleaned_transformed_CatgEncoded_v1_MinMaxScaler']['train']

print(train.target.value_counts())

train.target.value_counts().plot(kind='bar', title='Count (target)');
train_x = train.drop(['target'], axis=1)

train_y = train['target']

x_train, x_val, y_train, y_val = train_test_split(train_x, train_y,

                                                  test_size = .3,

                                                  random_state=12)



sm = SMOTE(random_state=12, sampling_strategy=0.9)#{1: 10, 0: 10})

x_train_res, y_train_res = sm.fit_sample(x_train, y_train)
y_train_res.value_counts().plot(kind='bar')
# train model

selection_model = XGBClassifier(learning_rate = 0.1,

                    n_estimators = 200,

                    max_depth = 5,

                    min_child_weight = 1,

                    gamma = 0,

                    subsample = 0.8,

                    colsample_bytree = 0.8,

                    objective = 'binary:logistic',

                    n_jobs = -1,

                    scale_pos_weight = 1,

                    verbose = False,

                    seed = 32)

     

selection_model.fit(x_train_res, y_train_res, eval_metric='logloss', verbose=False )        



# eval model    

train_pred_prob = selection_model.predict_proba(x_val)[:,1]    

log_loss_score = log_loss(y_val, train_pred_prob)   

print(log_loss_score)
# Train with the model that had the best result

selection_model = XGBClassifier(learning_rate = 0.1,

                    n_estimators = 200,

                    max_depth = 5,

                    min_child_weight = 1,

                    gamma = 0,

                    subsample = 0.8,

                    colsample_bytree = 0.8,

                    objective = 'binary:logistic',

                    n_jobs = -1,

                    scale_pos_weight = 1,

                    verbose = False,

                    seed = 32)



train = data['cleaned_transformed_CatgEncoded_v1_MinMaxScaler']['train']

train_x = train.drop(['target'], axis=1)

train_y = train['target']



final_model = selection_model.fit(train_x, train_y, eval_metric='logloss', verbose=False )        



# Test data for submission

test  = data['cleaned_transformed_CatgEncoded_v1_MinMaxScaler']['test']

test_x = test.drop(['ID'], axis=1)



# Performing predictions

test_pred_prob = final_model.predict_proba(test_x)[:,1]

submission = pd.DataFrame({'ID': test["ID"], 'PredictedProb': test_pred_prob.reshape((test_pred_prob.shape[0]))})

print(submission.head(10))
submission.to_csv('submission.csv', index=False)