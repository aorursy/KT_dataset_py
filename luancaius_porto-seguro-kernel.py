import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer, precision_score
from imblearn.over_sampling import SMOTE

from time import time
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
%matplotlib inline

# Setting plot for better images
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
pd.options.display.float_format = '{:.2f}'.format
rc={'savefig.dpi': 75, 'figure.autolayout': False, 'figure.figsize': [12, 8], 'axes.labelsize': 18,\
   'axes.titlesize': 18, 'font.size': 18, 'lines.linewidth': 2.0, 'lines.markersize': 8, 'legend.fontsize': 16,\
   'xtick.labelsize': 16, 'ytick.labelsize': 16}

sns.set(style='dark',rc=rc)
default_color = '#56B4E9'
colormap = plt.cm.cool
# Loading data and setting the timer to see how long it takes to run everything
start_init = time()
path = '../input/'
train = pd.read_csv(path+'train.csv',na_values=-1)
test = pd.read_csv(path+'test.csv',na_values=-1)
train.shape
train.head()
test.shape
# Using missingno library to check the missing values
def checkingMissingValues(dataset):         
    missingValueColumns = dataset.columns[dataset.isnull().any()].tolist()
    df_null = dataset[missingValueColumns] 
    msno.bar(df_null,figsize=(20,8),color=default_color,fontsize=18,labels=True)            
checkingMissingValues(train)
checkingMissingValues(test)
# Function that replaces the missing values with the mean of the column
def replacingMissingValues(dataset):
    col = dataset.columns
    for i in col:
        if dataset[i].isnull().sum() > 0:
            dataset[i].fillna(np.mean(dataset[i]), inplace=True)
    return dataset   
train = replacingMissingValues(train)
test = replacingMissingValues(test)
# Function to show the plot of the training dataset regarding the target column
def checkTarget(train):
    plt.figure(figsize=(15,5))
    ax = sns.countplot('target',data=train,color=default_color)
    for p in ax.patches:
        ax.annotate('{:.2f}%'.format(100*p.get_height()/len(train['target'])), (p.get_x()+ 0.3, p.get_height()+10000))
# Function to make the oversampling. It uses the imblearn.over_sampling library.
def oversampling(train):
    target = train['target']
    train = train.drop(['target'], axis=1)
    train['target'] = target
    features, target = SMOTE(n_jobs=-1, random_state=42).fit_sample(train.drop(['target'], axis=1), train['target'])
    target[target >= 0.5] = 1
    target[target < 0.5] = 0
    finalArray = np.column_stack((features, target))
    columns = train.columns.copy()
    train = pd.DataFrame(finalArray, columns=columns).reset_index(drop=True)
    return train
checkTarget(train)
train = oversampling(train)
checkTarget(train)
# Function that returns the value of prediction using normalized Gini coefficient.
def predict_labels_gini(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    y_pred = clf.predict(features)
    gini = 0
    try:
        gini = 2*roc_auc_score(target.values, y_pred)-1
    except ValueError:
        pass
    print("Gini score set: {:.4f}.".format(gini))
    return gini

# Function that trains the classifier and returns the prediction.
def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    clf.fit(X_train, y_train)

    return predict_labels_gini(clf, X_test, y_test)

def split_data(features):
    n_splits = 4
    folds = KFold(n_splits=n_splits, random_state=42)
    return enumerate(folds.split(features))

# Function that finds the best classifier from the array comparing the predict result.
def training_tuning(clfs, features, target, option):
    for i,clf in enumerate(clfs):
        bestModel = {}
        bestScore = 0
        bestFold = 0
        start_fold = time()
        for fold, (train_idx, test_idx) in split_data(features):
            print("\nFold ", fold)
            X_train = features.iloc[train_idx]
            y_train = target.iloc[train_idx]
            X_test = features.iloc[test_idx]
            y_test = target.iloc[test_idx]
            score = train_predict(clf, X_train, y_train, X_test, y_test)
            if score >= bestScore:
                bestScore = score
                bestModel = clf
                bestFold = fold
        end_fold = time()
        print('Training folds in {:.4f}'.format(end_fold - start_fold))
        if bestScore > 0:
            print('Tuning fold {} -> score {:.4f}'.format(bestFold, bestScore))
            bestTunnedModel = tuning(bestModel, features, target, bestScore)
            if bestTunnedModel == {}:
                submissionFile(bestModel, option, bestFold, clf.__class__.__name__)
            else:
                print('Best Tunned model')
                submissionFile(bestTunnedModel, option, bestFold, clf.__class__.__name__)

def normalized_gini(y_prob, y_actual):
    return 2 * roc_auc_score(y_prob, y_actual) - 1

def tuning(clf, features, target, score):
    print('Starting tunning')
    start_tuning = time()
    params={}
    if clf.__class__.__name__ == "DecisionTreeClassifier":
        params = {'min_samples_split': range(2, 202, 10)}

    if clf.__class__.__name__ == "LogisticRegression":
        params = {'class_weight': ['balanced'],
              'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'penalty': ['l1', 'l2']
              }

    if clf.__class__.__name__ == "SGDClassifier":
        params = {
            'max_iter':[250, 500, 1000],
            'loss': ['log'],
            }
    if params == {}:
        return clf
    # Normalized Gini Scorer
    gini_scorer = make_scorer(normalized_gini, greater_is_better=True)
    modelTunned = GridSearchCV(clf, params, scoring= gini_scorer, cv=4)
    modelTunned.fit(features, target)
    scoreTunned = predict_labels_gini(modelTunned, features, target)
    end_tuning = time()
    print("Finished tuning in {:.4f} seconds".format(end_tuning - start_tuning))
    if scoreTunned <= score:
        return {}
    return modelTunned
# Create submission file using the prediction from the tuned classifier
def submissionFile(clf, option, fold, clf_name):
    print('Creating submission file')
    sub = pd.DataFrame()
    sub['id'] = test['id']
    test_pred = pd.DataFrame(test, columns=train.drop(['id', 'target'], axis=1).columns)
    y_test_pred = clf.predict_proba(test_pred)[:, 1]
    sub['target'] = y_test_pred
    sub.to_csv('submit_{}_{}_{}.csv'.format(option, fold, clf_name), float_format='%.9f', index=False)
from sklearn.metrics import accuracy_score, log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

clfs = [
    DecisionTreeClassifier(),
    LogisticRegression(),
    # SGD used max_iter=100 for a fast decision if he was going to be tuned
    SGDClassifier(loss="log", max_iter=100)
]
options = [1,2,3,6]


for option in options:
    print("Starting Methodology {}".format(option))
    if option%2 == 0:
        print('Replacing values!')
        train = replacingMissingValues(train)
        test = replacingMissingValues(test)
    else:
        train.fillna(-1, inplace=True)
        test.fillna(-1, inplace=True)
    if option%3 == 0:
        print('Oversampling!')
        train = oversampling(train)

    target = train['target']
    features = train.drop(['id', 'target'], axis = 1)
    training_tuning(clfs, features, target, option)

end_init = time()
print("Finished in {:.4f} seconds".format(end_init - start_init))