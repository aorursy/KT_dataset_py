# Imports and setup

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #biblioteca gráfica para mostrar os gráficos

import warnings
warnings.filterwarnings('ignore')


DF = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import networkx as nx

#import plotly.express as px
#import plotly.figure_factory as ff
#from plotly.graph_objs import graph_objs
#import plotly.graph_objects as go
import matplotlib.pyplot as plt
#import seaborn as sns

import itertools
import time

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, Lasso, LogisticRegression
from sklearn.naive_bayes import GaussianNB
#from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from mlxtend.plotting import plot_decision_regions
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.
for c,d in zip(DF.columns, DF.dtypes):
    print("Column {}, type {}".format(c,d))
print("Number of missing values: {}.".format(DF.isna().sum().sum()))
print("Number of duplicated rows: {}.".format(DF.shape[0] - DF.drop_duplicates().shape[0]))
DF.drop_duplicates(inplace=True)
DF.describe()
# Analysing the features distribution on graphic way

def histogram(data, title, ax): #index
    n_bins = 30
    ax.hist(data, n_bins, density=True, histtype='bar')
    ax.legend(prop={'size': 8})
    ax.set_title(title)

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15,15))
for i in range(4):
    for j in range(3):
        idx_col = i*3+j
        if(idx_col >= DF.shape[1]):
            continue
        col = list(DF.columns)[idx_col]
        print(col)
        #axs[i][j] = histogram(DF[col])
        ax = axes[i][j]
        histogram(DF[col], col, ax)

fig.tight_layout()
plt.show()
corr = DF.corr()
#Plot Correlation Matrix using Matplotlib
plt.figure(figsize=(7, 5))
plt.imshow(corr, cmap='YlOrBr', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation='vertical')
plt.yticks(range(len(corr)), corr.columns);
plt.suptitle('Correlation between variables', fontsize=15, fontweight='bold')
plt.grid(False)
plt.show()
def correlation_pairs(df, threshold, sort=False):
    """
        Function to filter pair of features of a given DataFrame :df:
        that area correlated at least at :threshold:
    """
    pairs = []
    corr = df.corr()
    corr = corr.reset_index()
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if(j < i):
                col = corr.columns[j+1]
                corr_val = corr.loc[i][col]
                if(abs(corr_val) > threshold):
                    #print(i, j, corr.loc[i]['index'], col, corr_val)
                    pairs.append((corr.loc[i]['index'], col, corr_val))
    return pairs

correlation_pairs(DF, 0.3)
DF_qlt7 = DF[DF['quality'] < 7]
DF_qge7 = DF[DF['quality'] >= 7]
print(DF_qlt7.shape)
DF_qlt7.describe()
print(DF_qge7.shape)
DF_qge7.describe()
qlt7_stats = DF_qlt7.describe().loc[['mean', 'std']]
qge7_stats = DF_qge7.describe().loc[['mean', 'std']]
round(((qlt7_stats - qge7_stats) / qge7_stats) * 100, 2)
DF_qlt7_eq = DF_qlt7.sample(n=DF_qge7.shape[0], random_state=1)
qlt7_eq_stats = DF_qlt7_eq.describe().loc[['mean', 'std']]
round(((qlt7_eq_stats - qge7_stats) / qge7_stats) * 100, 2)
def boxplot(data, title, ax): #index
    green_diamond = dict(markerfacecolor='g', marker='D')
    ax.set_title(title)
    ax.boxplot(data, flierprops=green_diamond)

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15,15))
for i in range(4):
    for j in range(3):
        idx_col = i*3+j
        if(idx_col >= DF.shape[1]):
            continue
        col = list(DF.columns)[idx_col]
        print(col)
        #axs[i][j] = histogram(DF[col])
        ax = axes[i][j]
        boxplot(DF[col], col, ax)

fig.tight_layout()
plt.show()
# In order to detect and remove outliers we will use the zscore function of scipy.stats package
from scipy.stats import zscore
from functools import reduce

z = np.abs(zscore(DF))
threshold = 3 # our threshold will be 3 * std_dev
zmask = abs(z) > 3
zmask_per_line = [reduce(lambda curr, res : curr or res, zmask[i]) for i in range(len(zmask))]
print("Using Z-score to remove outliers we would remove ~ {} % of our data.".format(round(sum(zmask_per_line) / DF.shape[0] * 100), 2))

Q1 = DF.quantile(0.25)
Q3 = DF.quantile(0.75)
IQR = Q3 - Q1
iqrmask = (DF < (Q1 - 1.5 * IQR)) |(DF > (Q3 + 1.5 * IQR))
iqrmask_per_line = list(iqrmask.apply(lambda row : reduce(lambda curr, res : curr or res, row), axis=1))
print("Using IQR to remove outliers we would remove ~ {} % of our data.".format(round(sum(iqrmask_per_line) / DF.shape[0] * 100), 2))

zmask_per_line = [not z for z in zmask_per_line]

# So we use Z-score to reduce the data loss
DF = DF[zmask_per_line]
# First of all, we need to split data into two sets:
# X -> With all dependent variables
# y -> With target-feature

X = DF[DF.columns[:-1]]
y = DF[DF.columns[-1]]
from sklearn.preprocessing import StandardScaler

columns = X.columns

# Prepate the transformation function
scaler = StandardScaler().fit(X)
# Standardize data (mean=0, variance = 1)
X = pd.DataFrame(scaler.transform(X), columns=columns)
X.describe()
pca = PCA()
pca_result = pca.fit_transform(X)
var_exp = pca.explained_variance_ratio_

attr_x_var_exp = sorted(list(zip(X.columns, var_exp)), key=lambda x: x[1])
importances = [var_exp for _, var_exp in attr_x_var_exp]
attr_rank = [attr for attr, _ in attr_x_var_exp]

for attr, var_exp in attr_x_var_exp:
    print(attr, var_exp)

plt.title('Feature Importances')
plt.tight_layout()
plt.barh(range(len(importances)), importances, color='b', align='center')
plt.yticks(range(len(importances)), attr_rank, fontsize=25)
plt.xlabel('Relative Importance',fontsize=25)
plt.xticks(color='k', size=15)
plt.yticks(color='k', size=15)
plt.xlim([0.0, 1])
plt.show()
pca = PCA().fit(X)
plt.figure(figsize=(8,5))
ncomp = np.arange(1, np.shape(X)[1]+1)
plt.plot(ncomp, np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.xlabel('number of components', fontsize=15)
plt.ylabel('cumulative explained variance', fontsize=15);
plt.xticks(color='k', size=15)
plt.yticks(color='k', size=15)
plt.grid(True)
plt.show(True)
features_names = ['fixed acidity', 'volatile acidity']
X_ = X[features_names]
class_labels = np.unique(y)
#Plotting
all_colors = ['red', 'blue', 'orange', 'purple', 'green', 'yellow', 'black']
colors = all_colors[:len(class_labels)]
for i, c in enumerate(class_labels):
    ind = np.where(y == c)
    # mostra os grupos com diferentes cores
    plt.scatter(X_.loc[ind][X.columns[0]], X_.loc[ind][X.columns[1]], color = colors[i], label = c)
plt.legend()
plt.show()
def test_models(models, cv, X, y, scoring=None):
    for i in range(len(models)):
        print("Testing ", models[i]['name'])
        
        #if('multiclassTransformation' in models[i] and models[i]['multiclassTransformation']):
            # This line is required when using classification models for multi-class classification
        #    y_ = preprocessing.label_binarize(y, classes=list(y.unique()))
        #else:
        #    y_ = y

        y_ = y
        model = models[i]['model']
                 
        if('multiclassClassifier' in models[i]):
            multiclassClassifier = models[i]['multiclassClassifier']
            if(multiclassClassifier != None):
                #print(multiclassClassifier)
                model = multiclassClassifier(models[i]['model'])
        
        clf = GridSearchCV(model, models[i]['params'], cv=cv, scoring=scoring, verbose=2, n_jobs=-1)
        clf.fit(X, y_)
        models[i]['exec_time'] = (sum(clf.cv_results_['mean_fit_time']) * cv)
        models[i]['best_params'] = clf.best_params_
        models[i]['best_model'] = clf.best_estimator_ 
        models[i]['best_score'] = clf.best_score_
lb = preprocessing.LabelBinarizer()
y_encoded = lb.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.2, random_state = 42)
# ALL
# ---------------------------------------------------------------------
knn_params = {
    'n_neighbors' : list(range(6,50)),
    'weights' : ['uniform', 'distance'],
    'p' : [1, 2]
}
# ---------------------------------------------------------------------
svc_params = {
    'estimator__C' : [0.01, 0.1, 1, 10],
    'estimator__gamma' : ['auto', 'scale'],
    'estimator__class_weight' : [None, 'balanced'],
}
# ---------------------------------------------------------------------
dt_params = {
    'max_depth' : [1, 3, 5, 8, 13, 21, 34],
    'criterion' : ['gini', 'entropy'],
    'splitter' : ['best', 'random']
}
# ---------------------------------------------------------------------
gnb_params = {}
# ---------------------------------------------------------------------
rf_params = {
    'max_depth' : [1, 3, 5, 7, 11, 21],
    'n_estimators' : [3, 10, 20, 50, 100, 200],
    'max_features' : [2, 3, 5, 7, 9]
}
# ---------------------------------------------------------------------

Models = [
    {'name': "Dummy", 'model' : DummyClassifier(strategy='most_frequent', random_state=0), 'params' : {}, 'multiclassTransformation' : True, 'best_model' : None,'best_score' : 0, 'best_params' : None, 'exec_time' : 0.0},
    {'name': "KNN", 'model' : KNeighborsClassifier(), 'params' : knn_params, 'multiclassTransformation' : True, 'best_model' : None,'best_score' : 0, 'best_params' : None, 'exec_time' : 0.0},
    {'name': "DecisionTreeClassifier", 'model' : DecisionTreeClassifier(), 'params' : dt_params, 'multiclassTransformation' : True, 'best_model' : None,'best_score' : 0, 'best_params' : None, 'exec_time' : 0.0},
    {'name': "GaussianNB", 'model' : GaussianNB(), 'params' : gnb_params, 'multiclassTransformation' : True, 'multiclassClassifier' : OneVsRestClassifier, 'best_model' : None,'best_score' : 0, 'best_params' : None, 'exec_time' : 0.0},
    {'name': "SVC", 'model' : SVC(), 'params' : svc_params, 'multiclassTransformation' : True, 'multiclassClassifier' : OneVsRestClassifier, 'best_model' : None,'best_score' : 0, 'best_params' : None, 'exec_time' : 0.0},
    {'name': "RandomForestClassifier", 'model' : RandomForestClassifier(), 'params' : rf_params, 'multiclassTransformation' : True, 'best_model' : None,'best_score' : 0, 'best_params' : None, 'exec_time' : 0.0},
]

test_models(Models, 10, X_train, y_train, scoring=make_scorer(accuracy_score))
Models
from sklearn.metrics import confusion_matrix, classification_report

for model in Models:
    clf = model['best_model']
    name = model['name']
    
    predictions_clf = clf.predict(X_test)
    predictions_clf_decoded = lb.inverse_transform(predictions_clf)
    y_test_decoded = lb.inverse_transform(y_test)
    print(name ,classification_report(y_test_decoded, predictions_clf_decoded))
features_names = ['fixed acidity', 'volatile acidity']
X_ = X[features_names]

classifiers = [x['best_model'] for x in Models]
model_names = [x['name'] for x in Models]

# We save this variable to restore later
max_features = Models[5]['best_model'].max_features
# changing the best model of RandomForests to match with new number of features
Models[5]['best_model'].max_features = 2

for model in Models:
    clf = model['best_model']
    name = model['name']
    
    clf.fit(X_, y)
    
    plot_decision_regions(np.array(X_), np.array(y), clf=clf, legend=2)
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Decision Regions for ' + name)
    
    plt.show()
rf_params = {
    'max_depth' : [1, 3, 5, 7, 11, 21],
    'n_estimators' : [3, 10, 20, 50, 100, 200],
    'max_features' : [2, 3, 5, 7, 9]
}
# ---------------------------------------------------------------------

RFModel = [
    {'name': "RandomForestClassifier", 'model' : RandomForestClassifier(), 'params' : rf_params, 'multiclassTransformation' : True, 'best_model' : None,'best_score' : 0, 'best_params' : None, 'exec_time' : 0.0},
]

test_models(RFModel, 10, X, y, scoring=make_scorer(accuracy_score))
features_names = DF.columns
importances = RFModel[0]['best_model'].feature_importances_
indices = np.argsort(importances)
lmeas_order = []
for i in indices:
    lmeas_order.append(features_names[i])
plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), lmeas_order, fontsize=15)
plt.xlabel('Relative Importance',fontsize=15)
plt.xticks(color='k', size=15)
plt.yticks(color='k', size=15)
plt.show()
# =====================================================================================================
X = DF[DF.columns[:-1]]
y = DF[DF.columns[-1]].copy()

# =====================================================================================================

columns = X.columns
scaler = StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X), columns=columns)

# =====================================================================================================

y[DF[DF.columns[-1]] > 6.5] = 1
y[DF[DF.columns[-1]] <= 6.5] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# =====================================================================================================

# ALL
# ---------------------------------------------------------------------
knn_params = {
    'n_neighbors' : list(range(6,50)),
    'weights' : ['uniform', 'distance'],
    'p' : [1, 2]
}
# ---------------------------------------------------------------------
svc_params = {
    'estimator__C' : [0.01, 0.1, 1, 10],
    'estimator__gamma' : ['auto', 'scale'],
    'estimator__class_weight' : [None, 'balanced'],
}
# ---------------------------------------------------------------------
dt_params = {
    'max_depth' : [1, 3, 5, 8, 13, 21, 34],
    'criterion' : ['gini', 'entropy'],
    'splitter' : ['best', 'random']
}
# ---------------------------------------------------------------------
gnb_params = {}
# ---------------------------------------------------------------------
rf_params = {
    'max_depth' : [1, 3, 5, 7, 11, 21],
    'n_estimators' : [3, 10, 20, 50, 100, 200],
    'max_features' : [2, 3, 5, 7, 9]
}
# ---------------------------------------------------------------------

Models = [
    {'name': "Dummy", 'model' : DummyClassifier(strategy='most_frequent', random_state=0), 'params' : {}, 'multiclassTransformation' : True, 'best_model' : None,'best_score' : 0, 'best_params' : None, 'exec_time' : 0.0},
    {'name': "KNN", 'model' : KNeighborsClassifier(), 'params' : knn_params, 'multiclassTransformation' : True, 'best_model' : None,'best_score' : 0, 'best_params' : None, 'exec_time' : 0.0},
    {'name': "DecisionTreeClassifier", 'model' : DecisionTreeClassifier(), 'params' : dt_params, 'multiclassTransformation' : True, 'best_model' : None,'best_score' : 0, 'best_params' : None, 'exec_time' : 0.0},
    {'name': "GaussianNB", 'model' : GaussianNB(), 'params' : gnb_params, 'multiclassTransformation' : True, 'multiclassClassifier' : OneVsRestClassifier, 'best_model' : None,'best_score' : 0, 'best_params' : None, 'exec_time' : 0.0},
    {'name': "SVC", 'model' : SVC(), 'params' : svc_params, 'multiclassTransformation' : True, 'multiclassClassifier' : OneVsRestClassifier, 'best_model' : None,'best_score' : 0, 'best_params' : None, 'exec_time' : 0.0},
    {'name': "RandomForestClassifier", 'model' : RandomForestClassifier(), 'params' : rf_params, 'multiclassTransformation' : True, 'best_model' : None,'best_score' : 0, 'best_params' : None, 'exec_time' : 0.0},
]

test_models(Models, 10, X_train, y_train, scoring=make_scorer(accuracy_score))

# =====================================================================================================

for model in Models:
    clf = model['best_model']
    name = model['name']
    
    predictions_clf = clf.predict(X_test)
    print(name ,classification_report(y_test, predictions_clf))

features_names = ['fixed acidity', 'volatile acidity']
X_ = X[features_names]

classifiers = [x['best_model'] for x in Models]
model_names = [x['name'] for x in Models]

# We save this variable to restore later
max_features = Models[5]['best_model'].max_features
# changing the best model of RandomForests to match with new number of features
Models[5]['best_model'].max_features = 2

for model in Models:
    clf = model['best_model']
    name = model['name']
    
    clf.fit(X_, y)
    
    plot_decision_regions(np.array(X_), np.array(y), clf=clf, legend=2)
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Decision Regions for ' + name)
    
    plt.show()
rf_params = {
    'max_depth' : [1, 3, 5, 7, 11, 21],
    'n_estimators' : [3, 10, 20, 50, 100, 200],
    'max_features' : [2, 3, 5, 7, 9]
}
# ---------------------------------------------------------------------

RFModel = [
    {'name': "RandomForestClassifier", 'model' : RandomForestClassifier(), 'params' : rf_params, 'multiclassTransformation' : True, 'best_model' : None,'best_score' : 0, 'best_params' : None, 'exec_time' : 0.0},
]

test_models(RFModel, 10, X, y, scoring=make_scorer(accuracy_score))

# =====================================================================================================

features_names = DF.columns
importances = RFModel[0]['best_model'].feature_importances_
indices = np.argsort(importances)
lmeas_order = []
for i in indices:
    lmeas_order.append(features_names[i])
plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), lmeas_order, fontsize=15)
plt.xlabel('Relative Importance',fontsize=15)
plt.xticks(color='k', size=15)
plt.yticks(color='k', size=15)
plt.show()
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

X_for_regression = DF[list(DF.columns[:10]) + [DF.columns[-1]]]

columns = X_for_regression.columns

scaler = StandardScaler().fit(X)
X_for_regression = pd.DataFrame(scaler.transform(X), columns=columns)
y_for_regression = DF[DF.columns[10]]

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
p = 0.3 # fracao de elementos no conjunto de teste
x_train, x_test, y_train, y_test = train_test_split(X_for_regression, y_for_regression, test_size = p, random_state = 42)
lm = LinearRegression()
lm.fit(x_train, y_train)

y_pred = lm.predict(x_test)

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('R2 Coefficient: {} and MSE: {}', round(R2,2), round(mse,2))
import matplotlib.pyplot as plt

fig = plt.figure()
l = plt.plot(y_pred, y_test, 'bo')
plt.setp(l, markersize=10)
plt.setp(l, markerfacecolor='C0')

plt.ylabel("y", fontsize=15)
plt.xlabel("Prediction", fontsize=15)

# show original and predicted values
xl = np.arange(min(y_test), 1.2*max(y_test),(max(y_test)-min(y_test))/10)
yl = xl
plt.plot(xl, yl, 'r--')

plt.show(True)
np.random.seed(42)

vRDGRmse = []
vLASRmse = []
valpha = []
# varying values of alpha
for alpha in np.arange(1,30,1):
    
    ridge = Ridge(alpha = alpha, random_state=101, normalize=True)
    ridge.fit(x_train, y_train)             # Fit a ridge regression on the training data
    y_pred = ridge.predict(x_test)           # Use this model to predict the test data
    rmse = mean_squared_error(y_test, y_pred)
    vRDGRmse.append(rmse)
    
    lasso = Lasso(alpha = alpha, random_state=101, normalize=True) # normalize=True
    lasso.fit(x_train, y_train)             # Fit a lasso regression on the training data
    y_pred = lasso.predict(x_test)           # Use this model to predict the test data
    rmse = mean_squared_error(y_test, y_pred)
    vLASRmse.append(rmse)
    
    valpha.append(alpha)
    
plt.plot(valpha, vRDGRmse, '-ro')
plt.plot(valpha, vLASRmse, '-bo')
plt.xlabel("alpha", fontsize=15)
plt.ylabel("Mean Squared Error", fontsize=15)
plt.legend(['Ridge', 'Lasso'])
plt.show(True)