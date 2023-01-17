import os, logging, gc
from time import time
import pandas as pd
import numpy as np
import time

pd.set_option("display.max_columns", 50)

import warnings

warnings.filterwarnings(action='ignore')

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams 

%matplotlib inline
rcParams['figure.figsize'] = 15, 8

seed = 515
np.random.seed(seed)
def missing_values_table(df):
    #
    # Function to explore how many missing values (NaN) in the dataframe against its size
    # Args:
    #   df: the input dataframe for analysis
    # 
    # Return:
    #   mis_val_table_ren_columns: dataframe table contains the name of columns with missing data, # of missing values and % of missing against total
    #
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns and rows of " + str(df.shape[0]) + ".\n" "There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.\n")
    return mis_val_table_ren_columns


def read_data(filename, nrows=10):
    #
    # Function to read the csv file onto the panda dataframe
    # Args:
    #   filename: The name of csv file
    #   nrows: number of rows to be read. Default is 10 rows. None will read all rows
    #
    # Return:
    #  df: panda dataframe containing the data from csv file
    #
    if(os.path.isfile(filename)):
        print("\nReading file:: {}\n".format(filename))
        df = pd.read_csv(filename, sep = ',', nrows = nrows)
        df.columns = [x.lower() for x in df.columns]
        print("\n=======================================================================")
        print("Sample records: \n", df.head(2))
        print("\n=======================================================================")
        print("The data type: \n", df.columns.to_series().groupby(df.dtypes).groups)
        print("\n=======================================================================")
        print("Checking missing data (NaN): \n", missing_values_table(df))
        
    else:
        logging.warning("File is not existed")
        df = None
        
    return df


def one_way_tab (df, col):
    #
    # Function to compute one way table
    # Args:
    #   df: pandas dataframe
    #   col: column name to tabulate
    #
    # Return:
    #   df: the tabulate pandas of the outcome
    #
    sns.countplot(x = col, data = df)
    plt.show();
    df = pd.crosstab(index = df[col], columns = "count")
    df['percent'] = df/df.sum() * 100
    return df


data_file = "../input/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = read_data(data_file, nrows = None)
display(df.head(5))
one_way_tab(df, 'churn')
df[df.duplicated(['customerid'], keep=False)]
df['totalcharges'] = df['totalcharges'].replace(r'\s+', np.nan, regex=True)
df['totalcharges'] = pd.to_numeric(df['totalcharges'])
missing_values_table(df)
df[df.totalcharges.isnull()]
df.loc[df.totalcharges.isnull(), 'totalcharges'] = 0
sns.distplot(df.totalcharges)
plt.show();
def display_plot(df, col_to_exclude, object_mode = True):
    """ 
     This function plots the count or distribution of each column in the dataframe based on specified inputs
     @Args
       df: pandas dataframe
       col_to_exclude: specific column to exclude from the plot, used for excluded key 
       object_mode: whether to plot on object data types or not (default: True)
       
     Return
       No object returned but visualized plot will return based on specified inputs
    """
    n = 0
    this = []
    
    if object_mode:
        nrows = 4
        ncols = 4
        width = 20
        height = 20
    
    else:
        nrows = 2
        ncols = 2
        width = 14
        height = 10
    
    
    for column in df.columns:
        if object_mode:
            if (df[column].dtypes == 'O') & (column != col_to_exclude):
                this.append(column)
                
                
        else:
            if (df[column].dtypes != 'O'):
                this.append(column)
     
    
    fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(width, height))
    for row in range(nrows):
        for col in range(ncols):
            if object_mode:
                sns.countplot(df[this[n]], ax=ax[row][col])
                
            else:
                sns.distplot(df[this[n]], ax = ax[row][col])
            
            ax[row,col].set_title("Column name: {}".format(this[n]))
            ax[row, col].set_xlabel("")
            ax[row, col].set_ylabel("")
            n += 1

    plt.show();
    return None

display_plot(df, 'customerid', object_mode = True)
display_plot(df, 'customerid', object_mode = False)
pd.crosstab(index = df["phoneservice"], columns = df["multiplelines"])
pd.crosstab(index = df["internetservice"], columns = df["streamingtv"])
def convert_no_service (df):
    col_to_transform = []
    for col in df.columns:
        if (df[col].dtype == 'O') & (col != 'customerid'):
            if len(df[df[col].str.contains("No")][col].unique()) > 1:
                col_to_transform.append(col)
    
    print("Total column(s) to transform: {}".format(col_to_transform))
    for col in col_to_transform:
        df.loc[df[col].str.contains("No"), col] = 'No'
        
    return df

df = convert_no_service(df)
display_plot(df, 'customerid', object_mode = True)
df.gender = df.gender.map(dict(Male=1, Female=0))
display(df.gender.value_counts())
def encode_yes_no (df, columns_to_encode):
    for col in columns_to_encode:
        df[col] = df[col].map(dict(Yes = 1, No = 0))
        
    return df

encode_columns = []
for col in df.columns:
    keep = np.sort(df[col].unique(), axis = None)
    
    if ("Yes" in keep) & ("No" in keep):
        encode_columns.append(col)

del keep
print("Encode Columns Yes/No: {}".format(encode_columns))
        
    
df = encode_yes_no(df, encode_columns)
display(df.head(5))
df = pd.get_dummies(df, columns = ['internetservice', 'contract', 'paymentmethod'], prefix = ['ISP', 'contract', 'payment'])
display(df.head(5))
df2 = df.drop('customerid', axis = 1, inplace = False)
df2.columns = df2.columns.str.replace(" ", "_")
df2.corr()['churn'].sort_values(ascending=False)
corr = df2.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(16, 10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.6, cbar_kws={"shrink": .5})
plt.show();
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support
import pickle
import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import make_scorer

X = df2.drop('churn', axis = 1, inplace = False)
y = df2['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed)
print("Training target distribution:\n{}".format(y_train.value_counts()))
print("\nTesting target distribution:\n{}".format(y_test.value_counts()))

def xgb_f1(y, t):
    #
    # Function to evaluate the prediction based on F1 score, this will be used as evaluation metric when training xgboost model
    # Args:
    #   y: label
    #   t: predicted
    #
    # Return:
    #   f1: F1 score of the actual and predicted
    #
    t = t.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y]   # change the prob to class output
    return 'f1', f1_score(t, y_bin)


def plot_evaluation_metric (y_true, y_prob):
    #
    # Function to plot the evaluation metric (cumulative gain, lift chart, precision and recall) on the screen
    # Args:
    #   y_true: array of y true label
    #   y_prob: array of y predicted probability (outcome of predict_proba() function)
    #
    # Return:
    #   None
    #
    skplt.metrics.plot_cumulative_gain(y_true, y_prob)
    plt.show();
    skplt.metrics.plot_precision_recall(y_true, y_prob)
    plt.show();
    skplt.metrics.plot_lift_curve(y_true, y_prob)
    plt.show();
    return 


def print_evaluation_metric (y_true, y_pred):
    #
    # Function to print out the model evaluation metrics
    # Args:
    #   y_true: array of y true label
    #   y_pred: array of y predicted class
    #
    # Return:
    #   None
    #
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F-score: {}".format(fscore))
    print("Support: {}".format(support))
    return 


def get_confusion_matrix (y_true, y_pred, save=0, filename="this.csv"):
    #
    # Function to print out the confusion matrix on screen as well as print to csv file, if enabled
    # Args:
    #   y_true: array of y true label
    #   y_pred: array of y prediction
    #   save: to enable the write to csv file (default = 0)
    #   filename: the name of the file to be saved (default = this.csv)
    #
    # Return:
    #   None
    #
    from sklearn.metrics import confusion_matrix
    get_ipython().magic('matplotlib inline')
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred),
                      columns = ['Predicted False', 'Predicted True'],
                      index = ['Actual False', 'Actual True']
                      )
    display(cm)
    if(save):
        cm.to_csv(filename, index = True)
    
    return 


def my_plot_roc_curve (y_true, y_prob, filename="img.png", dpi = 200):
    #
    # Function to plot the ROC curve by computing fpr and tpr as well as save the plot to file
    # Args:
    #   y_true: array of y true label
    #   y_prob: the output of y probability prediction (outcome for predict_proba() function)
    #   filename: the name of the file to be saved
    #   dpi: the resolution of the figure
    # Return:
    #   None
    #
    fpr, tpr, threshold = roc_curve(y_true, y_prob[:, 1])
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.plot(fpr, tpr, 'b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    fig.savefig(filename, dpi = dpi)
    return


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return


classifiers = [
    KNeighborsClassifier(n_jobs = 4),
    RandomForestClassifier(n_jobs = 4),
    XGBClassifier(n_jobs = 4)
]

# iterate over classifiers
for item in classifiers:
    classifier_name = ((str(item)[:(str(item).find("("))]))
    print (classifier_name)
    
    # Create classifier, train it and test it.
    clf = item
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print ("Score: ", round(score,3),"\nF1 score: ", round(f1_score(y_test, pred), 3), "\n- - - - - ", "\n")
    
param_grid = {
    'silent': [False],
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.001, 0.01, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [0.5, 1.0, 3.0],
    'gamma': [0, 0.25, 0.5, 1.0],
    'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
    'n_estimators': [50, 100, 150],
    'scale_pos_weight': [1, 1.5, 2],
    'max_delta_step': [1, 2, 3]
}

clf = XGBClassifier(objective = 'binary:logistic')
fit_params = {'eval_metric': 'logloss',
              'early_stopping_rounds': 10,
              'eval_set': [(X_test, y_test)]}

rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=50,
                            n_jobs=4, verbose=2, cv=5,
                            fit_params=fit_params,
                            scoring= 'f1_macro', refit=True, random_state=seed)


print("Randomized search..")
search_time_start = time.time()
rs_clf.fit(X_train, y_train)
print("Randomized search time:", time.time() - search_time_start)

best_score = rs_clf.best_score_
best_params = rs_clf.best_params_
print("Best score: {}".format(best_score))
print("Best params: ")
for param_name in sorted(best_params.keys()):
    print('%s: %r' % (param_name, best_params[param_name]))
best_xgb = XGBClassifier(objective = 'binary:logistic',
                         colsample_bylevel = 0.7,
                         colsample_bytree = 0.8,
                         gamma = 1,
                         learning_rate = 0.15,
                         max_delta_step = 3,
                         max_depth = 4,
                         min_child_weight = 1,
                         n_estimators = 50,
                         reg_lambda = 10,
                         scale_pos_weight = 1.5,
                         subsample = 0.9,
                         silent = False,
                         n_jobs = 4
                        )

best_xgb.fit(X_train, y_train, eval_metric = xgb_f1, eval_set = [(X_train, y_train), (X_test, y_test)], 
             early_stopping_rounds = 20)
xgb.plot_importance(best_xgb, max_num_features = 15)
plt.show();
y_pred = best_xgb.predict(X_test)
y_prob = best_xgb.predict_proba(X_test)
print_evaluation_metric(y_test, y_pred)
get_confusion_matrix (y_test, y_pred, save=0, filename="this.csv")
my_plot_roc_curve (y_test, y_prob, filename="ROC.png", dpi = 200)
plot_evaluation_metric (y_test, y_prob)
from sklearn.metrics import classification_report
ev = classification_report(y_test, y_pred, target_names = ['Not Churn', 'Churn'])
print(ev)
from xgboost import plot_tree
import graphviz

plot_tree(best_xgb, num_trees = 0)
fig = plt.gcf()
fig.set_size_inches(300, 100)
fig.savefig('tree.png')
y_all_prob = best_xgb.predict_proba(X)
df['churn_prob'] = y_all_prob[:, 1]
sns.distplot(df['churn_prob'])
plt.show();
df[['customerid', 'churn', 'churn_prob']].head(10)
import shap
shap.initjs()

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_train)
shap.force_plot(explainer.expected_value, shap_values, X_train)
shap.summary_plot(shap_values, X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")
shap.dependence_plot("ISP_Fiber_optic", shap_values, X_train, interaction_index="monthlycharges")

shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], link="logit")