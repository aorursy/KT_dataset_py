# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns #for better and easier plots

%matplotlib inline



# Ignore useless warnings (see SciPy issue #5998)

import warnings

warnings.filterwarnings(action="ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
app_df = pd.read_csv("../input/application_train.csv")
app_df.shape
app_df.head()
app_test_df = pd.read_csv("../input/application_test.csv")
app_test_df.shape
#let's create a function to check for null values, calculate the percentage relative to the total size

#only shows the null values present in the dataset

def missing_values_calculate(trainset): 

    nulldata = (trainset.isnull().sum() / len(trainset)) * 100

    nulldata = nulldata.drop(nulldata[nulldata == 0].index).sort_values(ascending=False)

    ratio_missing_data = pd.DataFrame({'Ratio' : abs(nulldata)})

    return ratio_missing_data.head(30)
def remove_missing_columns(train, test, threshold = 90): #threshold is set by default at 90%

    # Calculate missing stats for train and test (remember to calculate a percent!)

    train_miss = pd.DataFrame(train.isnull().sum())

    train_miss['percent'] = 100 * train_miss[0] / len(train)

    

    test_miss = pd.DataFrame(test.isnull().sum())

    test_miss['percent'] = 100 * test_miss[0] / len(test)

    

    # list of missing columns for train and test

    missing_train_columns = list(train_miss.index[train_miss['percent'] > threshold])

    missing_test_columns = list(test_miss.index[test_miss['percent'] > threshold])

    

    # Combine the two lists together

    missing_columns = list(set(missing_train_columns + missing_test_columns))

    

    # Print information

    print('There are %d columns with greater than %d%% missing values.' % (len(missing_columns), threshold))

    

    # Drop the missing columns and return

    train = train.drop(columns = missing_columns)

    test = test.drop(columns = missing_columns)

    

    return train, test
missing_values_calculate(app_df)
app_df.info()
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype



        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
app_df.head()
(app_df['DAYS_BIRTH'] / -365).describe()
app_df['DAYS_EMPLOYED'].describe()
app_df[app_df['DAYS_EMPLOYED'] >= 300000].describe() 
app_df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

app_test_df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True) #imputing the testing set as well...
corr = app_df.corr()['TARGET'].sort_values(ascending=False)



corr.head(10) #looking at the first 10 positively correlated
corr.tail(10).sort_values() #now looking at the first 10 negatively correlated
plt.style.use('dark_background')

plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time

sns.kdeplot(app_df.loc[app_df['TARGET'] == 0, 'DAYS_BIRTH'] * -1/ 365, label = 'target == 0')



# KDE plot of loans which were not repaid on time

sns.kdeplot(app_df.loc[app_df['TARGET'] == 1, 'DAYS_BIRTH'] * -1/ 365, label = 'target == 1')



# Labeling of plot

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
# Extract the EXT_SOURCE variables and show correlations

ext_data = app_df[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

ext_data_corrs = ext_data.corr()



# Heatmap of correlations

sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

plt.title('Correlation Heatmap');
plt.figure(figsize = (10, 12))



# iterate through the sources

for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):

    

    # create a new subplot for each source

    plt.subplot(3, 1, i + 1)

    # plot repaid loans

    sns.kdeplot(app_df.loc[app_df['TARGET'] == 0, source], label = 'target == 0')

    # plot loans that were not repaid

    sns.kdeplot(app_df.loc[app_df['TARGET'] == 1, source], label = 'target == 1')

    

    # Label the plots

    plt.title('Distribution of %s by Target Value' % source)

    plt.xlabel('%s' % source); plt.ylabel('Density');

    

plt.tight_layout(h_pad = 2.5)

    
app_df.shape
#let's use the Imputer to fill the NAN values with the median value

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,MinMaxScaler,Imputer, RobustScaler



#imputing all NaN value

pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")), 

        #('scale', MinMaxScaler(feature_range = (0, 1))),

        ('robustScaler', RobustScaler()),

])
from sklearn.metrics import precision_score, recall_score, accuracy_score,confusion_matrix,classification_report,f1_score, roc_auc_score

import time #implementing in this function the time spent on training the model

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,cross_val_score,train_test_split, KFold

import gc



nfolds = 5

folds = KFold(n_splits=nfolds, shuffle=True,random_state=42)





#Generic function for making a classification model and accessing performance:

def classification_model(train, train_labels, test_set, pipeline, params={}, fold=folds, plot_confusion_matrix=False, model=None, GridSearch=False, plot_features_importances=False):

    

    time_start = time.perf_counter() #start counting the time

    #creating our validation set out of the training set and labels provided

    X_train, x_val, y_train, y_val = train_test_split(train, train_labels, test_size=0.1, random_state=42)

    X_train = pipeline.fit_transform(X_train) #fiting and transforming the dataset using the pipeline provided

    x_val = pipeline.fit_transform(x_val)

    

    test_sub = np.zeros(test_set.shape[0])

    test_set = pipeline.fit_transform(test_set)

    

    predict_val = np.zeros(train.shape[0])

    score = {}

    

    if model != None: grid_model = GridSearchCV(model, params,verbose=1, cv=3) #initializing the grid search model



    if GridSearch:

        grid_model.fit(X_train, y_train)

        score_grid = grid_model.best_score_

        

        #predicting using the model that has been trained above

        

        predict_val = grid_model.predict(x_val)

        score['Accuracy'] = (accuracy_score(y_val, predict_val))

        score['Precision'] = (precision_score(y_val, predict_val))

        score['F1 score'] = (f1_score(y_val, predict_val))

        score['ROC AUC'] = (roc_auc_score(y_val, predict_val))

        

        print("Model Report")



        print("Accuracy: "+ str(score["Accuracy"]))

        print("Precision: "+ str(score["Precision"]))

        print("F1 score: "+ str(score["F1 score"]))

        print("ROC AUC: "+ str(score["ROC AUC"]))

        print('\n')

        

        print("         -------Classification Report----------")

        print(classification_report(y_val, predict_val))

    

        test_sub = grid_model.predict(test_set) 

        

    else:

        model = lgb.LGBMClassifier(**params, n_estimators = 5000, nthread = 4, n_jobs = -1)



        for n, (index, val_index) in enumerate(folds.split(train)):

            

            print('Starting Fold number: %d' %n)

            X, X_val = train.values[index], train.values[val_index]

            Y, Y_val = train_labels[index], train_labels[val_index]

            X = pipeline.fit_transform(X)

            X_val = pipeline.fit_transform(X_val)

            

            model.fit(X, Y, 

                    eval_set=[(X, Y), (X_val, Y_val)],

                    verbose=1000, early_stopping_rounds=200)

            

            y_pred_valid = model.predict(X_val)

            test_temp = model.predict(test_set, num_iteration=model.best_iteration_)

            test_sub += test_temp               

            if score == {}:

                score['Accuracy'] = accuracy_score(Y_val, y_pred_valid)

                score['Precision']= precision_score(Y_val, y_pred_valid)

                score['F1 score'] = f1_score(Y_val, y_pred_valid)

                score['ROC AUC'] = roc_auc_score(Y_val, y_pred_valid)

            else:

                score['Accuracy'] += accuracy_score(Y_val, y_pred_valid)

                score['Precision'] += precision_score(Y_val, y_pred_valid)

                score['F1 score'] += f1_score(Y_val, y_pred_valid)

                score['ROC AUC'] += roc_auc_score(Y_val, y_pred_valid)

                        

        test_sub /= nfolds

                        

        print("Model Report")



        print("Accuracy(avg across folds): "+ str(score["Accuracy"]/nfolds))

        print("Precision(avg across folds): "+ str(score["Precision"]/nfolds))

        print("F1 score(avg across folds): "+ str(score["F1 score"]/nfolds))

        print("ROC AUC(avg across folds): "+ str(score["ROC AUC"]/nfolds))

        print('\n')



        

    #################### PLOTTING FEATURES IMPORTANCE ####################

    

    # Sort features according to importance

    if plot_features_importances:

        if GridSearch:

            # Extract feature importances

            feature_importances = pd.DataFrame({'feature': list(train.columns), 'importance': grid_model.best_estimator_.feature_importances_})

        else:

            feature_importances = pd.DataFrame({'feature': list(train.columns), 'importance': model.feature_importances_})

        

        feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index()



        # Normalize the feature importances to add up to one

        feature_importances['importance_normalized'] = feature_importances['importance'] / feature_importances['importance'].sum()



        # Make a horizontal bar chart of feature importances

        plt.figure(figsize = (10, 6))

        ax = plt.subplot()



        # Need to reverse the index to plot most important on top

        ax.barh(list(reversed(list(feature_importances.index[:15]))), 

                feature_importances['importance_normalized'].head(15), 

                align = 'center', edgecolor = 'k')



        # Set the yticks and labels

        ax.set_yticks(list(reversed(list(feature_importances.index[:15]))))

        ax.set_yticklabels(feature_importances['feature'].head(15))



        # Plot labeling

        plt.xlabel('Normalized Importance'); plt.title('Feature Importances')

    

    #################### PLOTTING CONFUSION MATRIX #######################

    

    if plot_confusion_matrix:

        fig, ax = plt.subplots(figsize=(8,8)) #setting the figure size and ax

        mtx = confusion_matrix(y_val, predict_val)

        sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  cbar=True, ax=ax) #create a heatmap with the values of our confusion matrix

        plt.ylabel('true label')

        plt.xlabel('predicted label')



    

    time_end = time.perf_counter() #end of counting the time

    

    total_time = time_end-time_start #total time spent during training and cross_validation

    

    print("Amount of time spent during training the model and cross validation: %4.3f seconds" % (total_time))

    # Clean up memory

    gc.enable()

    del model, X_train, x_val, y_train, y_val,score, total_time, time_end, time_start,predict_val,test_set

    gc.collect()

                        

    return test_sub
bureau_df = pd.read_csv('../input/bureau.csv')#loading the dataset

bureau_df.shape #checking the shape of our dataset
bureau_df.head()
num_of_previous_credits = bureau_df.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'}) 

#grouping by the client's ID and counting the number of credits using the ID bureau. then,renaming the column to something more informative

num_of_previous_credits.head()
app_df = app_df.merge(num_of_previous_credits, on = 'SK_ID_CURR', how = 'left')

app_df['previous_loan_counts'] = app_df['previous_loan_counts'].fillna(0) #filling all the clients that doesn't have past loans with 0
#getting only the relevant columns that we would like to see the correlation, in this case, getting previous columns plus the new feature

relevant_columns = app_df[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'previous_loan_counts']]

corr = relevant_columns.corr()



# Heatmap of correlations

sns.heatmap(corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

plt.title('Correlation Heatmap');
## CREDITS TO WILL KOERSEHN ##

# Plots the disribution of a variable colored by value of the target

def kde_target(var_name, df):

    

    # Calculate the correlation coefficient between the new variable and the target

    corr = df['TARGET'].corr(df[var_name])

    

    # Calculate medians for repaid vs not repaid

    avg_repaid = df.ix[df['TARGET'] == 0, var_name].median()

    avg_not_repaid = df.ix[df['TARGET'] == 1, var_name].median()

        

    # Plot the distribution for target == 0 and target == 1

    sns.kdeplot(df.ix[df['TARGET'] == 0, var_name], label = 'TARGET == 0')

    sns.kdeplot(df.ix[df['TARGET'] == 1, var_name], label = 'TARGET == 1')

    

    # label the plot

    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)

    plt.legend();

    

    # print out the correlation

    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))

    # Print out average values

    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)

    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)

    
plt.figure(figsize = (10, 12))

for i,column in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'previous_loan_counts']):

    plt.subplot(5,1, i+1)

    kde_target(column, app_df)

plt.tight_layout(h_pad = 2.5)
## CREDITS TO WILL KOERSEHN ##



def agg_numeric(df, group_var, df_name):

    """Aggregates the numeric values in a dataframe. This can

    be used to create features for each instance of the grouping variable.

    

    Parameters

    --------

        df (dataframe): 

            the dataframe to calculate the statistics on

        group_var (string): 

            the variable by which to group df

        df_name (string): 

            the variable used to rename the columns

        

    Return

    --------

        agg (dataframe): 

            a dataframe with the statistics aggregated for 

            all numeric columns. Each instance of the grouping variable will have 

            the statistics (mean, min, max, sum; currently supported) calculated. 

            The columns are also renamed to keep track of features created.

    

    """

    # Remove id variables other than grouping variable

    for col in df:

        if col != group_var and 'SK_ID' in col:

            df = df.drop(columns = col)

            

    group_ids = df[group_var]

    numeric_df = df.select_dtypes('number')

    numeric_df[group_var] = group_ids



    # Group by the specified variable and calculate the statistics

    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()



    # Need to create new column names

    columns = [group_var]



    # Iterate through the variables names

    for var in agg.columns.levels[0]:

        # Skip the grouping variable

        if var != group_var:

            # Iterate through the stat names

            for stat in agg.columns.levels[1][:-1]:

                # Make a new column name for the variable and stat

                columns.append('%s_%s_%s' % (df_name, var, stat))



    agg.columns = columns

    return agg
bureau_grouped = agg_numeric(bureau_df.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name= 'bureau_df')

bureau_grouped.head()
bureau_grouped.shape
app_df = app_df.merge(bureau_grouped, on = 'SK_ID_CURR', how = 'left')

app_df.head()
app_df.shape
## CREDITS TO WILL KOERSEHN ##



def count_categorical(df, group_var, df_name):

    """Computes counts and normalized counts for each observation

    of `group_var` of each unique category in every categorical variable

    

    Parameters

    --------

    df : dataframe 

        The dataframe to calculate the value counts for.

        

    group_var : string

        The variable by which to group the dataframe. For each unique

        value of this variable, the final dataframe will have one row

        

    df_name : string

        Variable added to the front of column names to keep track of columns



    

    Return

    --------

    categorical : dataframe

        A dataframe with counts and normalized counts of each unique category in every categorical variable

        with one row for every unique value of the `group_var`.

        

    """

    

    # Select the categorical columns

    categorical = pd.get_dummies(df.select_dtypes('object'))



    # Make sure to put the identifying id on the column

    categorical[group_var] = df[group_var]



    # Groupby the group var and calculate the sum and mean

    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])

    

    column_names = []

    

    # Iterate through the columns in level 0

    for var in categorical.columns.levels[0]:

        # Iterate through the stats in level 1

        for stat in ['count', 'count_norm']:

            # Make a new column name

            column_names.append('%s_%s_%s' % (df_name, var, stat))

    

    categorical.columns = column_names

    

    return categorical

bureau_grouped_categories = count_categorical(bureau_df, group_var = 'SK_ID_CURR', df_name= 'bureau_df')

bureau_grouped_categories.head()
app_df = app_df.merge(bureau_grouped_categories, on = 'SK_ID_CURR', how = 'left')

app_df.head()
app_df.shape
bureau_df_balance = pd.read_csv('../input/bureau_balance.csv')

bureau_df_balance.head()
previous_ap_df = pd.read_csv('../input/previous_application.csv')

previous_ap_df.head()
reduce_mem_usage(previous_ap_df) #reducing memory usage
previous_agg = agg_numeric(previous_ap_df, group_var = 'SK_ID_CURR', df_name = 'previous')

print(previous_ap_df.shape, previous_agg.shape)
previous_count = count_categorical(previous_ap_df, group_var = 'SK_ID_CURR', df_name='previous')

print(previous_ap_df.shape, previous_count.shape)
# Counts of each type of status for each previous loan, it's a categorical feature

bureau_balance_counts = count_categorical(bureau_df_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_df_balance')

bureau_balance_counts.head()
# Calculate value count statistics for each `SK_ID_CURR`

bureau_balance_agg = agg_numeric(bureau_df_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_df_balance')

bureau_balance_agg.head()
# Dataframe grouped by the loan

bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')



# Merge to include the SK_ID_CURR

bureau_by_loan = bureau_df[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on = 'SK_ID_BUREAU', how = 'left')



# Aggregate the stats for each client

bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client')
bureau_balance_by_client.head()
app_df = app_df.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')

app_df = app_df.merge(previous_count, on = 'SK_ID_CURR', how = 'left')

app_test_df = app_test_df.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')

app_test_df = app_test_df.merge(previous_count, on = 'SK_ID_CURR', how = 'left')
# Merge with the monthly information grouped by client

app_df = app_df.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')

#applying all the changes that has been made to the training set, to the testing set.

app_test_df = app_test_df.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')

app_test_df = app_test_df.merge(bureau_grouped, on = 'SK_ID_CURR', how = 'left')

app_test_df = app_test_df.merge(bureau_grouped_categories, on = 'SK_ID_CURR', how = 'left')

app_test_df = app_test_df.merge(num_of_previous_credits, on = 'SK_ID_CURR', how = 'left')
print(app_df.shape, app_test_df.shape)
#getting dummies for our dataset

app_df = pd.get_dummies(app_df)

app_test_df = pd.get_dummies(app_test_df)

print(app_df.shape, app_test_df.shape)
gc.enable()

del bureau_df,previous_agg,previous_ap_df,previous_count, num_of_previous_credits, bureau_df_balance, bureau_grouped,bureau_grouped_categories, bureau_balance_agg,bureau_by_loan, bureau_balance_by_client

gc.collect()
app_df = reduce_mem_usage(app_df) #using the function to reduce the amount of memory used by our dataframe(function has been defined at the beggining of the notebook)

app_test_df = reduce_mem_usage(app_test_df)
app_df.head() #taking a look at how our dataframe looks now.
missing_values_calculate(app_df) #using our function to calculate the missing values
app_df, app_test_df = remove_missing_columns(app_df,app_test_df, threshold=73)
corr = app_df.corr()

corr['TARGET'].sort_values(ascending=False).head(20)
corr['TARGET'].sort_values(ascending=False).tail(20)
#creating our Y, our target

train_labels = app_df['TARGET'].copy()

app_df_no_ids = app_df.drop(['SK_ID_CURR','DAYS_ID_PUBLISH'], axis=1)

app_df_no_ids, app_test_df_no_ids = app_df_no_ids.align(app_test_df, join = 'inner', axis = 1)
from sklearn.linear_model import LogisticRegression



param_grid = {'C': [0.0001], 'multi_class': ['multinomial'],  

              'penalty': ['l1'],'solver': ['saga'], 'tol': [0.1] }



log_reg = LogisticRegression()



# Train on the training data

test_log_reg = classification_model(app_df_no_ids[:30000:], train_labels[:30000:],test_set=app_test_df_no_ids,pipeline=pipeline ,params=param_grid, GridSearch=True, model=log_reg)
from sklearn.ensemble import RandomForestClassifier



param_grid_random = {'n_estimators': [100]}

# Make the random forest classifier

random_forest = RandomForestClassifier(random_state = 42, verbose = 1, n_jobs = -1)

prediction_lgr = classification_model(app_df_no_ids[:30000:], train_labels[:30000:], test_set=app_test_df_no_ids, pipeline=pipeline ,params=param_grid_random, GridSearch=True, model=random_forest, plot_features_importances=True)
import lightgbm as lgb



params_lgb = { 

              "learning_rate": [0.05],

              "reg_alpha": [0.1],

              "reg_lambda": [0.1],

              "subsample": [0.8],

                'class_weight': 'balanced'

}
submission_prediction_LGB = classification_model(app_df_no_ids, train_labels, pipeline=pipeline ,params=params_lgb, GridSearch=False, plot_features_importances=True,test_set=app_test_df_no_ids)
submission_prediction_LGB = submission_prediction_LGB > 0.8 #as the prediction we got is an average across all the folds, we have a percentage for each target, I am gonna get only percentages greater than 80% here for target = 1
sub = pd.read_csv("../input/sample_submission.csv")

sub['TARGET'] = submission_prediction_LGB.astype('int8')

sub.to_csv('lgb_new_features.csv', index=False)
sub.head()
sub["TARGET"].value_counts()