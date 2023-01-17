import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_curve
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
import missingno as msno
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
import gc
import warnings
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
import xgboost as xgb
import contextlib
import sys
from io import StringIO
%matplotlib inline
mpl.rcParams['figure.figsize'] = (10, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
pd.set_option('display.float_format', lambda x: '%.3f' % x)
train_dataset_path = '../input/malicious-server-hack/Train.csv'
test_dataset_path = '../input/malicious-server-hack/Test.csv'
raw_train_df = pd.read_csv(train_dataset_path)
raw_train_df.head()
raw_test_df = pd.read_csv(test_dataset_path)
raw_test_df.head()
# Assign class column name to a variable 
class_variable = "MALICIOUS_OFFENSE"
### Check data types of the column
raw_train_df.dtypes
raw_train_df['DATE'] = pd.to_datetime(raw_train_df['DATE'])
raw_train_df['day'] = raw_train_df['DATE'].dt.day
raw_train_df['month'] = raw_train_df['DATE'].dt.month
# Drop DATE Column
raw_train_df.drop(columns=["DATE"],inplace=True)
# Set INCIDENT_ID to index
raw_train_df= raw_train_df.set_index('INCIDENT_ID')
raw_train_df.describe()
def missing_values_table(input_df):
    """
    Returns the number of missing values in each column (if it has any missing values) and percentage of missing values.

    Parameters
    ----------
    input_df: pd.DataFrame
        The dataframe that whose missing data information is required 

    Returns
    -------
    mis_val_table_ren_columns: pd.DataFrame
        Returns a dataframe containing columns and missing data information

    """
    # Total missing values
    mis_val = input_df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * input_df.isnull().sum() / len(input_df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Values Missing'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Values Missing', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(input_df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns
train_missing= missing_values_table(raw_train_df)
train_missing
test_missing= missing_values_table(raw_test_df)
test_missing
msno.bar(raw_train_df)
#sorted by X_12
sorted_df = raw_train_df.sort_values('X_12')
msno.matrix(sorted_df)
#sorted by X_12
sorted_test_df = raw_test_df.sort_values('X_12')
msno.matrix(sorted_test_df)
def impute_missing_data(input_df, columns):
    """
    Imputes the missing data in given column
    
    Parameters
    ----------
    input_df: pd.DataFrame
        The dataframe that whose column is to be imputed

    columns: list
        List containing names of the columns that needs to be imputed


    Returns
    -------
    result_df: pd.DataFrame
        Returns the dataframe with imputed values.
    """
    knn_imputer = KNNImputer(n_neighbors=5, weights="uniform")
    for column_name in columns:
        input_df[column_name] = knn_imputer.fit_transform(input_df[[column_name]])
    
    result_df = input_df.copy()

    return result_df
## We have missing data in X_12
train_df = impute_missing_data(raw_train_df, ['X_12'])
def scale_dataframe(input_df, columns):
    """
        Scales the given columns of input dataframe
        
        Parameters
        ----------
        input_df: pd.DataFrame
            The dataframe that has to be scaled
        
        columns: list
            List containing names of the columns that needs to be scaled
            
        
        Returns
        -------
        result_df: pd.DataFrame
            Returns the normalized dataframe.
    """
    scaler = StandardScaler()
    non_scale_columns = list(filter(lambda col : col not in columns, input_df.columns))
    normalized_df = pd.DataFrame(scaler.fit_transform(input_df[columns]), columns=columns)
    result_df = pd.concat(normalized_df, input_df[non_scale_columns])
    
    return result_df
# Correlation with all variables with MULTIPLE_OFFENSE
train_df.corr().nlargest(18, 'MALICIOUS_OFFENSE')['MALICIOUS_OFFENSE'].index
f,ax = plt.subplots(figsize=(16, 16))
high_to_low_col_index = train_df.corr().nlargest(19, class_variable)[class_variable].index
sns.heatmap(train_df[high_to_low_col_index].corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)
neg, pos = np.bincount(train_df[class_variable])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n    Negative: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total, neg, 100*neg/total))
X = train_df.loc[:, train_df.columns != class_variable]
Y = train_df.loc[:, train_df.columns == class_variable]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33,random_state=8)

print('Training labels shape:', X_train.shape)
print('Test labels shape:', X_test.shape)
print('Training features shape:', y_train.shape)
print('Test features shape:', y_test.shape)
X_resample,y_resample=SMOTE(sampling_strategy=0.85).fit_sample(X_train, y_train)
X_resample.head()
y_resample.head()
print('Resampled Training features shape:', X_resample.shape)
print('Resampled Training labels shape:', y_resample.shape)
def generate_metrics(labels, predictions):
    """
        Calculates the metrics like accuracy, weighted recall, weighted precision
        and F1 score.
        
        Parameters
        ----------
        labels: 1d-array
            True values of the class variable
        
        predictions: 1d-array
            Predictions by the model
            
        
        Returns
        -------
        Does not return anything
    
    """
    ac = accuracy_score(labels,predictions)
    f_score = f1_score(labels,predictions)
    recall = recall_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    print('Accuracy is: ', ac)
    print('Recall is:', recall )
    print('Precision is:', precision)
    print('F1 score is: ', f_score)
    
def plot_roc(name, labels, predictions, **kwargs):
    """
        This helper function plots the receiver operating characteristic curve. One of the best metrics 
        to evaluate a model other than F1 score and Kappa score.
        
        Parameters
        ----------
        labels: 1d-array
            True values of the class variable
        
        predictions: 1d-array
            Predictions by the model
            
        
        Returns
        -------
        Does not return anything
    
    """
    fp, tp, _ = roc_curve(labels, predictions)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

def plot_cm(labels, predictions, p=0.5):
    """
        This helper function plots the confusion matrix
        
        Parameters
        ----------
        labels: 1d-array
            True values of the class variable
        
        predictions: 1d-array
            Predictions by the model
        
        p: Float
            The thresold value
            
        
        Returns
        -------
        Does not return anything
    
    """
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))
if globals().get("XG_base",None) is not None: 
    del XG_base
    gc.collect()
    
XG_base = XGBClassifier(random_state=17)
XG_base.fit(X_resample, y_resample.values.ravel())

baseline_model_xg = XG_base.fit(X_resample, y_resample.values.ravel())
baseline_test_predictions = baseline_model_xg.predict(X_test)
baseline_train_predictions = baseline_model_xg.predict(X_resample)
generate_metrics(y_test, baseline_test_predictions)
plot_cm(y_test,baseline_test_predictions)
plot_roc("Baseline Train", y_resample, baseline_train_predictions, color=colors[1])
plot_roc("Baseline Test", y_test, baseline_test_predictions, color=colors[1], linestyle='--')
plt.legend(loc='lower right')

# save base model
baseline_model_xg.save_model("baseline_model")
@contextlib.contextmanager
def capture():
    """
    Captures the output and writes to logfile
    """
    olderr, oldout = sys.stderr, sys.stdout
    try:
        out=[StringIO(), StringIO()]
        sys.stderr,sys.stdout = out
        yield out
    finally:
        sys.stderr,sys.stdout = olderr,oldout
        out[0] = out[0].getvalue().splitlines()
        out[1] = out[1].getvalue().splitlines()
def load_data():
    """
    Loads a copy of train and test data.
    """
    train = X_resample.copy()
    train_labels = y_resample.copy()
    print('\n Shape of raw train data:', train.shape)

    return train, train_labels
def XGB_CV(
          max_depth,
          gamma,
          min_child_weight,
          max_delta_step,
          subsample,
          colsample_bytree,
          n_estimators
         ):
    """
    This is the Cross-validation function with given parameters. 
    We will optimize this function using Bayesian Optimization
    
    Parameters
    ----------
    The parameters of the XGBoost that I want to optimize.
    
    Returns
    ----------
    Returns cv_score to caller function
    """

    global AUCPRbest
    global ITERbest

#
# Define all XGboost parameters
#

    paramt = {
              'booster' : 'gbtree',
              'max_depth' : int(max_depth),
              'gamma' : gamma,
              'eta' : 0.1,
              'objective' : 'binary:logistic',
              'nthread' : 4,
              'silent' : True,
              'eval_metric': 'aucpr',
              'subsample' : max(min(subsample, 1), 0),
              'colsample_bytree' : max(min(colsample_bytree, 1), 0),
              'min_child_weight' : min_child_weight,
              'max_delta_step' : int(max_delta_step),
              'seed' : 1001,
              'n_estimators' : int(n_estimators),
              'random_state': 17
              }

    folds = 5
    cv_score = 0

    print("\n Search parameters (%d-fold validation):\n %s" % (folds, paramt), file=log_file )
    log_file.flush()

    xgbc = xgb.cv(
                    paramt,
                    dtrain,
                    num_boost_round = 20000,
                    stratified = True,
                    nfold = folds,
                    early_stopping_rounds = 100,
                    metrics = 'aucpr',
                    show_stdv = True
               )


    with capture() as result:
        warnings.filterwarnings('ignore')
        val_score = xgbc['test-aucpr-mean'].iloc[-1]
        train_score = xgbc['train-aucpr-mean'].iloc[-1]
        print(' Stopped after %d iterations with train-auc = %f val-auc = %f ( diff = %f ) train-gini = %f val-gini = %f' % ( len(xgbc), train_score, val_score, (train_score - val_score), (train_score*2-1),
    (val_score*2-1)) )
        if ( val_score > AUCPRbest ):
            AUCPRbest = val_score
            ITERbest = len(xgbc)

    return (val_score*2) - 1
log_file = open('AUCPR-5fold-XGB-run-01-v1-full.log', 'a')
AUCPRbest = -1.
ITERbest = 0

# Load data set and target values
train, target = load_data()

dtrain = xgb.DMatrix(train, label = target)
#set the lower and upper searching bounds
bounds = {'max_depth': (2, 12), 'gamma': (0.001, 10.0), 'min_child_weight': (0, 20),
          'max_delta_step': (0, 10),'subsample': (0.4, 1.0),'colsample_bytree' :(0.4, 1.0),'n_estimators': (20,300)
                                    }
XGB_BO = BayesianOptimization(XGB_CV, bounds )
# This might take a couple of minutes to run

print('-'*130)
print('-'*130, file=log_file)
log_file.flush()

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    XGB_BO.maximize(init_points=2, n_iter=7, acq='ei', xi=0.0)
best_cv_score = XGB_BO.max
best_cv_score
print(best_cv_score)
optimal_params = best_cv_score["params"]
optimal_params["max_depth"] = int(optimal_params["max_depth"])
optimal_params["n_estimators"] = int(optimal_params["n_estimators"])
if globals().get("XG_optimal",None) is not None: 
    del XG_optimal
    
XG_optimal = XGBClassifier(random_state=17,**optimal_params)
XG_optimal.fit(X_resample, y_resample.values.ravel())
optimal_model_xg = XG_optimal.fit(X_resample, y_resample.values.ravel())
optimal_test_predictions = optimal_model_xg.predict(X_test)
optimal_train_predictions = optimal_model_xg.predict(X_resample)
generate_metrics(y_test, optimal_test_predictions)

plot_cm(y_test,optimal_test_predictions)
plot_roc("Optimal Train", y_resample, optimal_train_predictions, color=colors[1])
plot_roc("Optimal Test", y_test, optimal_test_predictions, color=colors[1], linestyle='--')
plt.legend(loc='lower right')
# save optimal model
XG_optimal.save_model("BO_optimal_model")
#This might take a couple of minutes to run

rfecv_xg = RFECV(estimator=XG_optimal, step=1, cv=6, scoring='recall_weighted')
rfecv_xg = rfecv_xg.fit(X_resample, y_resample.values.ravel())
print('Optimal number of features :', rfecv_xg.n_features_)
best_features = X_resample.columns[rfecv_xg.support_].tolist()
print('Best features :', best_features)
sorted(rfecv_xg.grid_scores_,reverse=True)
plt.figure(figsize=(15,15))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv_xg.grid_scores_) + 1), rfecv_xg.grid_scores_)
plt.grid()
plt.show()
x_train_rfecv_xg = rfecv_xg.transform(X_resample)
x_test_rfecv_xg = rfecv_xg.transform(X_test)
x_train_rfecv_xg.shape
#Fitting rfecv or XG_optimal is same, one can verify by using 'get_params()' attribute in both.
if globals().get("XG_optimal",None) is not None: 
    del XG_optimal
    
XG_optimal = XGBClassifier(random_state=17,**optimal_params)
rfecv_model_xg = XG_optimal.fit(x_train_rfecv_xg, y_resample.values.ravel())
rfecv_test_predictions = rfecv_model_xg.predict(x_test_rfecv_xg)
rfecv_train_predictions = rfecv_model_xg.predict(x_train_rfecv_xg)
generate_metrics(y_test, rfecv_test_predictions)





plot_cm(y_test,rfecv_test_predictions)

plot_roc("Optimal Train", y_resample, rfecv_train_predictions, color=colors[1])
plot_roc("Optimal Test", y_test, rfecv_test_predictions, color=colors[1], linestyle='--')
plt.legend(loc='lower right')
# Save Final Model
rfecv_model_xg.save_model("final_model")
def inference(saved_model_name, features):
    """
    This function runs the inference code
    
    Parameters
    ------------
    saved_model_name: XGBoost Model
        Name of the saved model
    features: list 
        List of the features selected by RFECV
    
    Returns
    -----------
    submission_df: pd.DataFrame
        Returns the dataframe with predicted values
    """
    # Load testdata
    submission_df = pd.read_csv('../input/malicious-server-hack/Test.csv')
    #df_test.fillna('0',inplace=True)

    # Pre-process Data
    
    # Create Month and Date Columns
    submission_df['DATE'] = pd.to_datetime(submission_df['DATE'])
    submission_df['day'] = submission_df['DATE'].dt.day
    submission_df['month'] = submission_df['DATE'].dt.month

    # Set "INCIDENT ID" as index
    submission_df.set_index('INCIDENT_ID', inplace= True)

    #Select features based on RFECV 
    submission_df_ = submission_df[features].copy()

    #Impute the missing values in column X_12
    if 'X_12' in submission_df.columns:
        submission_df_ = impute_missing_data(submission_df_, ['X_12'])
    # Load Model
    inference_model = XGBClassifier()
    inference_model.load_model(saved_model_name)
    
    # Prediction
    predictions = inference_model.predict(submission_df_.to_numpy())
    prediction_df=pd.DataFrame(predictions,columns=[class_variable])
    
    # Create Submission Dataframe    
    index_df =pd.DataFrame(submission_df.index)
    submission_df = pd.concat([index_df,prediction_df], axis=1)
    
    return submission_df
# Save to submission.csv
final_submission_df = inference("final_model", best_features)
final_submission_df.to_csv("final_model_submission.csv",index=False)
print(final_submission_df.MALICIOUS_OFFENSE.value_counts())
print(final_submission_df.MALICIOUS_OFFENSE.value_counts()/len(final_submission_df))
