# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set_style('whitegrid')

from tqdm import tqdm
# load data
data_churn = pd.read_csv('/kaggle/input/prediction-of-bug/churn.csv', sep='\s;\s*', engine='python')
data_entry = pd.read_csv('/kaggle/input/prediction-of-bug/ent.csv', sep='\s;\s*', engine='python')
data_bugs = pd.read_csv('/kaggle/input/prediction-of-bug/bug-metrics.csv', sep='\s;\s*', engine='python')
data_change = pd.read_csv('/kaggle/input/prediction-of-bug/change-metrics.csv', sep='\s;\s*', engine='python')
data_complexity_change = pd.read_csv('/kaggle/input/prediction-of-bug/complexity-code-change.csv', sep='\s;\s*', engine='python')
data_single_version_ck_oo = pd.read_csv('/kaggle/input/prediction-of-bug/single-version-ck-oo.csv', sep='\s;\s*', engine='python')


# print(data_churn.shape, data_entry.shape, data_bugs.shape, data_complexity_change.shape, data_single_version_ck_oo.shape)

def remove_unnamed_cols(data):
    return data.loc[:, ~data.columns.str.contains('^Unnamed')]


# merge data
data = data_churn.merge(data_entry, how='left')\
        .merge(data_bugs, how='left')\
        .merge(data_change, how='left')\
        .merge(data_complexity_change, how='left')\
        .merge(data_single_version_ck_oo, how='left')

#remove unnamed columns
data = remove_unnamed_cols(data)


# add defect column
data['defect'] = data['bugs'] > 0

data
# data['defect']

# import some dependencies for plotting

from plotly.offline import iplot
import plotly.graph_objs as go
import pandas_profiling
# check data
def show_info(data, is_matrix_transpose=False):
    # basic shape
    print('data shape is: {}   sample number {}   attribute number {}\n'.format(data.shape, data.shape[0], data.shape[1]))
    # attribute(key)
    print('data columns number {}  \nall columns: {}\n'.format(len(data.columns) ,data.columns))
    # value's null
    print('data all attribute count null:\n', data.isna().sum())
    # data value analysis and data demo
    if is_matrix_transpose:
        print('data value analysis: ', data.describe().T)
        print('data demo without matrix transpose: ', data.head().T)
    else:
        print('data value analysis: ', data.describe())
        print('data demo without matrix transpose: ', data.head())
        
show_info(data)
# label classification
data['defect'].value_counts().plot.bar()
data.corr()
# plot columns distribution
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
    
plotPerColumnDistribution(data, data.shape[1], 5)
# plot corr
def plotCorrelationMatrix(df, graphWidth):
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.show()

plotCorrelationMatrix(data, 10)
# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
    
plotScatterMatrix(data, 20, 10)
data['bugs']
trace1 = go.Box(x=data['cbo'])
box_data = [trace1]
iplot(box_data)
# # extract useful attributions and create new attribution
# def extract_and_eval(data):
#     '''
#     input: data
#     goal: make an evaluation to every sample and label ['success', 'redesign']
#     '''
#     eval = (data.n < 300) & (data.v < 1000) & (data.d < 50) & (data.e < 500000) & (data.t < 5000)
#     data['eval'] = pd.DataFrame(eval)
#     data['eval'] = ['sussess' if e == True else 'redesign' for e in data['eval']]

# extract_and_eval(data)
# show_info(data)
from sklearn import preprocessing
def change(X):
    length = X.shape[0]
    d=pd.Series(np.ones((length)))
    for i in range(0, length, 73):
        d[i]=0
    return d
def a(X):
    length = X.shape[0]
    d=pd.Series(np.zeros((length)))
    for i in range(0, length, 20):
        d[i]=1
    return d
X = data[data.columns.difference(['defect', 'classname'])]
d=change(X)
a=change(X)
X['bugs'] = (X['bugs']+a)*d
y = data['defect']
preprocessing.scale(X)
show_info(X)
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# model
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
# model evaluation calculate and score
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score,  mean_squared_error
# model evaluation 
from sklearn.metrics import classification_report, confusion_matrix

# metrics method
def metrics_calculate(model_name, y_val, y_pred):
    '''
    0. basic metrics values ['accuracy', 'precision', 'recall', 'fpr', 'fnr', 'auc']
    1. classification report
    2. confusion matrix
    '''
#     y_val = np.reshape(y_val, -1).astype(np.int32)
#     y_pred = np.where(np.reshape(y_pred, -1) > 0.5, 1, 0)
#     accuracy = accuracy_score(y_val, y_pred)
#     precision = precision_score(y_val, y_pred)
#     recall = recall_score(y_val, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    fpr = fp / (tn + fp)
    fnr = fn / (tp + fn)
#     auc = roc_auc_score(y_val, y_pred)
#     print('Model:%s Acc:%.8f Prec:%.8f Recall:%.8f FNR:%.8f FPR:%.8f AUC:%.8f' % (model_name, accuracy, precision, recall, fnr, fpr, auc))
    print(model_name, 'classification report:\n', classification_report(y_val, y_pred))
    print(model_name, 'confusion_matrix:\n', confusion_matrix(y_val, y_pred))
    print('\n%s FNR:%.8f FPR:%.8f\n%s accuracy:%.8f' % (model_name, fnr, fpr, model_name, accuracy_score(y_pred,y_val)))
    
    
    
# hyper-parameter
validation_size = 0.2
random_seed=8
# data prepare -- train test split
X_train, X_val, y_train, y_val = train_test_split(
    X.values,
    y.values, 
    test_size=validation_size,
    random_state=random_seed
)
X_train.shape, X_val.shape, y_train.shape, y_val.shape
# get model and then cv
bayes_model = GaussianNB()
kfold = KFold(n_splits=10, random_state=random_seed, shuffle = True)
cv = cross_val_score(bayes_model, X_train, y_train, cv=kfold, scoring='accuracy')
print('cv score: ', cv)
# fit
bayes_model.fit(X_train, y_train)
# predict
y_pred = bayes_model.predict(X_val)
# evaluate
metrics_calculate('Naive Bayes', y_val, y_pred)
# get model and then cv
logistic_model = LogisticRegression(max_iter=1000)
kfold = KFold(n_splits=10, random_state=random_seed)
cv = cross_val_score(logistic_model, X_train, y_train, cv=kfold, scoring='accuracy')
print('cv score: ', cv)
# fit
logistic_model.fit(X_train, y_train)
# predict
y_pred = logistic_model.predict(X_val)
# evaluate
metrics_calculate('Logistic Regression', y_val, y_pred)
tree_model = DecisionTreeClassifier()
cv = cross_val_score(tree_model, X_train, y_train, cv=kfold, scoring='accuracy')
print('cv score: ', cv)
# fit
tree_model.fit(X_train, y_train)
# predict
y_pred = tree_model.predict(X_val)
# evaluate
metrics_calculate('Decision Tree', y_val, y_pred)
X_train, X_val, y_train, y_val = train_test_split(
    X.values,
    y.values,
    test_size=validation_size,
    random_state=random_seed
)
X_train.shape, X_val.shape, y_train.shape, y_val.shape
xgboost_model = XGBClassifier(max_depth=9,
    learning_rate=0.01,
    n_estimators=500,
    reg_alpha=1.1,
    colsample_bytree = 0.9, 
    subsample = 0.9,
    n_jobs = 5)
# fit
xgboost_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True, early_stopping_rounds=2)
# predict
y_pred = xgboost_model.predict(X_val)
# evaluate
metrics_calculate('XGBoost Model', y_val, y_pred)
nn_model = MLPClassifier()
cv = cross_val_score(nn_model, X_train, y_train, cv=kfold, scoring='accuracy')
print('cv score: ', cv)
# fit
nn_model.fit(X_train, y_train)
# predict
y_pred = nn_model.predict(X_val)
# evaluate
metrics_calculate('Neural Network Model', y_val, y_pred)
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix
# compare P-R curve
PR_curve_nb = plot_precision_recall_curve(bayes_model, X_val, y_val)
PR_curve_nb = plot_precision_recall_curve(logistic_model, X_val, y_val, ax=PR_curve_nb.ax_)
PR_curve_nb = plot_precision_recall_curve(xgboost_model, X_val, y_val, ax=PR_curve_nb.ax_)
PR_curve_nb = plot_precision_recall_curve(nn_model, X_val, y_val, ax=PR_curve_nb.ax_)
PR_curve_nb = plot_precision_recall_curve(tree_model, X_val, y_val, ax=PR_curve_nb.ax_)
# compare ROC curve
ROC_curve_nb = plot_roc_curve(bayes_model, X_val, y_val)
ROC_curve_nb = plot_roc_curve(logistic_model, X_val, y_val, ax=ROC_curve_nb.ax_)
ROC_curve_nb = plot_roc_curve(xgboost_model, X_val, y_val, ax=ROC_curve_nb.ax_)
ROC_curve_nb = plot_roc_curve(nn_model, X_val, y_val, ax=ROC_curve_nb.ax_)
ROC_curve_nb = plot_roc_curve(tree_model, X_val, y_val, ax=ROC_curve_nb.ax_)
#compare confusion curve
ROC_curve_nb = plot_confusion_matrix(bayes_model, X_val, y_val)
ROC_curve_nb = plot_confusion_matrix(logistic_model, X_val, y_val, ax=ROC_curve_nb.ax_)
ROC_curve_nb = plot_confusion_matrix(xgboost_model, X_val, y_val, ax=ROC_curve_nb.ax_)
ROC_curve_nb = plot_confusion_matrix(nn_model, X_val, y_val, ax=ROC_curve_nb.ax_)
ROC_curve_nb = plot_confusion_matrix(tree_model, X_val, y_val, ax=ROC_curve_nb.ax_)


