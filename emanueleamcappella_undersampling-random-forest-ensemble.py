import pandas as pd
import math
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import re
import matplotlib.pyplot as plt
import os
from random import randint, seed

from sklearn.utils import resample
import sklearn.model_selection
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing
from tqdm import tqdm_notebook

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, auc

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#utility
from IPython.display import display, HTML
import itertools

pd.options.display.float_format = '{:,.3f}'.format
path='../input/creditcardfraud/'
df = pd.read_csv(path + 'creditcard.csv')
# create minutes and drop seconds
df['Hours'] = df.Time/3600
df.drop(['Time'], axis=1, inplace=True)
df.head()
# Barplots
def make_bar(var, cutoff, head, colour):
    p = pd.cut(var, bins = cutoff)
    p.value_counts(sort=False, normalize= True).plot(kind='bar', color= colour)
    plt.title(head)
    plt.xlabel('Categories')
    plt.ylabel('Frequency')
    
f, axes = plt.subplots(figsize = (15, 10))
plt.subplot(121) 
ax = sns.distplot(df.Hours)
ax.set_title('Distribution of Hours')
plt.subplot(122)
make_bar(df['Hours'], [df.Hours.min(), 12, 24, 36, df.Hours.max()], 'Hours Barplot', 'blue')
plt.figure(figsize=(15, 10))
make_bar(df['Amount'], [df.Amount.min(), 1, 10, 100, 500, df.Amount.max()], 'Money Amount', 'green')
# correlation between features
# whitegrid
sns.set_style('whitegrid')
#compute correlation matrix...
corr_matrix=df.drop('Class', axis=1).corr(method='pearson')
# delete correlations between components (all the columns starting with V), because by definitions they are uncorrelated (corr=0)
cmatrix_reduced = corr_matrix.loc[:, ~corr_matrix.columns.str.contains('V.*')]

plt.figure(figsize=(40, 30))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cmatrix_reduced, cmap=cmap, linewidth=0.5, square=True, vmin=-1, vmax=1, annot=True, cbar_kws={"shrink": .5})
percentage = round(len(df[df.Class == 1])/len(df)*100, 3)
print(f'Percentage of frauds: {percentage}%')
#separazione dati training e test in modo stratificato
test_size = 0.20
#Balancing parameters for building our bagging datasets
perc_1 = 0.05
perc_0 = 1 - perc_1

y = df.Class.to_frame()
df_train, df_test = sklearn.model_selection.train_test_split(df, test_size=test_size, random_state=123, stratify=y)
X_train = df_train.drop('Class', axis=1)

X_test = df_test.drop('Class', axis=1)
y_train = df_train.Class
y_test = df_test.Class

print('Number of cases for X_train: ', len(X_train))
print('Number of cases for X_test: ', len(X_test))
# undersampling for unbalanced dataset
def do_UNDERSAMPLING(df, perc_1 = 0.05, perc_0 = 0.95, random_state=123):
    df_negative = df[df.Class==0]
    df_positive = df[df.Class==1]
    size_1= df_positive.shape[0]
    size_0 = df_positive.shape[0] * perc_0/perc_1
    
    # Downsample majority class
    df_negative_downsampled = resample(df_negative, 
                                     replace=False,    # sample without replacement
                                     n_samples=math.ceil(size_0),     # to match minority class
                                     random_state=random_state) # for reproducible results

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_negative_downsampled, df_positive])
    X_downsampled = df_downsampled.drop('Class',axis=1)
    y_downsampled = df_downsampled.Class
    return df_downsampled, X_downsampled, y_downsampled


#variable to choose the balancing method for the test set
test_bil = 'Complete' #0.01% fraud out of the total data = completed vs 5% frauds = Balanced 

#undersampling if we want to Balance the test set
if test_bil == 'Balanced':
    X_test['Class'] = y_test
    df_temp , X_test, y_test = do_UNDERSAMPLING(X_test, perc_1, perc_0, random_state=123)
#scale to unitary variance and zero mean
def scale(X_train):
    numerics=['float64']
    temp = X_train.copy()
    numeric_feat=temp.select_dtypes(include=numerics)
    scaled_cols=numeric_feat.columns.values
    scaler = preprocessing.StandardScaler().fit(numeric_feat)
    temp.loc[:,scaled_cols]=scaler.fit_transform(temp.loc[:,scaled_cols])
    return temp, scaler, scaled_cols

#scaling of our train dataset, mean = 0 and unitary variance
scaled_df_train, std_scaler, scaled_cols = scale(X_train)
# scale also x test
X_test.loc[:,scaled_cols]=std_scaler.transform(X_test.loc[:,scaled_cols])
n_models = 6 #n of RF in the ensemble

# vectors containing training datasets for every model
x_trains = []
y_trains = []

X_train['Class'] = y_train
 
sampling = 'UNDERSAMPLING'

# seed selection to get reproducible results
seed(a=123)
seeds = [randint(0,999) for i in range(n_models)]

#creating dataset using undersampling
for i in range(n_models):
    df_temp , X_temp, y_temp = do_UNDERSAMPLING(X_train, perc_1, perc_0, random_state=seeds[i])
    x_trains.append(X_temp)
    y_trains.append(y_temp)
print('Dummy results if you always predict one:')
print('\tPrecision: {}'.format(precision_score(np.ones(len(y_test)), y_test)))
print('\tRecall: {}'.format(recall_score(np.ones(len(y_test)), y_test)))
print('\tF1-score: {}'.format(recall_score(np.ones(len(y_test)), y_test)))
imp_thr = 12 #n of variables to keep when applying the RF feature selection

def rf_importance(X_train, y_train, imp_thr=imp_thr):
    forest = RandomForestClassifier(n_estimators=250, random_state=123)
    num_df = X_train.select_dtypes('float64','int')
    str_df = X_train.select_dtypes('object')
    forest.fit(num_df, y_train)
    importances = forest.feature_importances_
    data = {'variables': num_df.columns.values, 'importance': importances}
    importance_df = pd.DataFrame.from_dict(data)
    importance_df.sort_values(by='importance', ascending=False, inplace=True)
    
    if imp_thr > 0: 
        importance_df=importance_df.head(imp_thr)
    
    num_df = num_df[np.asarray(importance_df.variables)]
    X_train=pd.concat([str_df, num_df], axis=1)
    return  X_train, importance_df
# evaluation metrics
scores_functions = {
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score
}

corr_thr = 0.80 #correlation threshold
scores = list(scores_functions.keys())

#configuration parameters
fixed_params = {
    'p_test_size': test_size,
    'p_train_size':  1 - test_size,
    'p_s_perc_1': perc_1,
    'p_s_perc_0': perc_0,
    'p_corr_thr': corr_thr,
    'p_importance_thr': imp_thr,
    'p_sampling': sampling,
    'p_bilanciam_test': test_bil,
    'p_n_models': n_models
}

#n of folds for cross-validation
cv_splits = 5
model_name = 'final_test'
# the final output will be saved on an Excel File named filename+model_name
model = RandomForestClassifier

#gridsearch configuration for the training set
tuned_parameters = [{'bootstrap': [False],
 'max_depth': [10,20],
 'max_features': ['log2'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 3, 5],
 'n_estimators': [10,20,25,30]}]

df_res = pd.DataFrame()

#inside these arrays we will store the results of the cv from the n-models  
best_models = []
predictions = []
predictions_proba = []

print('* * * Starting model fitting')
for i in range(n_models):
    
    #each model reads a dedicated part of the total train dataset
    x_t = x_trains[i].copy()
    y_t = y_trains[i].copy()
    #RF for feature selection on the train set
    x_train, _ = rf_importance(x_t, y_t, imp_thr=imp_thr)
    x_train = x_train.select_dtypes('float64','int')
    y_train = y_trains[i]
    
    # corresponding feature selection on the test set
    x_test = X_test[x_train.columns.tolist()]
    
    # gridsearch for the tuning of the parameters 
    print('* * * Starting gridsearch for the {}-th model...'.format(i))
    clf = GridSearchCV(model(random_state=123), tuned_parameters, cv=StratifiedKFold(n_splits=cv_splits, random_state=123),
                       scoring=scores, n_jobs=-1, refit='precision', return_train_score=True, verbose=True)

    clf.fit(x_train, y_train)
    print('* * * Gridsearch end')

    #best model
    best_est = clf.best_estimator_
    #best configuration
    best_par = clf.best_params_
    
    #for each model gridsearch identifies the best model, which is then stored in the dedicated array
    best_models.append(best_est)
    
    dict_res = {} #dictionary containing all the configuration parameters
    dict_res.update(fixed_params)
    
    #creating dataframe to save it on excel later on
    for i in tqdm_notebook(range(len(clf.cv_results_['params']))):
        params = clf.cv_results_['params'][i]
        #saving CV results of the best model
        if params == best_par:
            
            #saving all the CV metrics
            for score in scores:
                dict_res['cv_mean_test_{}'.format(score)] = clf.cv_results_['mean_test_{}'.format(score)][i]
                dict_res['cv_std_test_{}'.format(score)] = clf.cv_results_['std_test_{}'.format(score)][i]
                dict_res['cv_mean_train_{}'.format(score)] = clf.cv_results_['mean_train_{}'.format(score)][i]
                dict_res['cv_std_train_{}'.format(score)] = clf.cv_results_['std_train_{}'.format(score)][i]
            #saving model parameters
            for key in params.keys():
                dict_res['model_param_{}'.format(key)] = params[key]
    
    #prediction using the best classifiers out of the gridsearch
    y_pred = best_est.predict(x_test)
    
    predictions.append(y_pred) #predictions 0/1
    predictions_proba.append(best_est.predict_proba(x_test)[:,1]) #probability predictions
    
    df_res = df_res.append(dict_res, ignore_index=True)
print('* * * Models fitting over')
# Choose the method of aggregation (majority voting\mean probability scores)
aggregation_method = 'PROBA' # or 'VOTING' as alternative

th = 0.65 # probability threshold used for the voting section

if aggregation_method == 'VOTING':   
    df_res['aggregation'] = 'VOTING'
    y_sum = np.zeros(len(y_test))
    for i in range(n_models):
        y_sum = y_sum + predictions[i]
        
    final_pred = (y_sum >= np.ceil(n_models/2)).astype(int)
    
elif aggregation_method == 'PROBA':
    df_res['aggregation'] = 'PROBA_{}'.format(th)
    mean_prob = np.mean(predictions_proba, axis=0)
    final_pred = (mean_prob > th).astype(int)

# save test results
for score in scores:
        metric = scores_functions[score]
        df_res['test_{}'.format(score)] = metric(y_true=y_test, y_pred=final_pred)

df_res.head()
#confusion matrix function
def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
    """
   This function prints and plots the confusion matrix.
   Normalization can be applied by setting `normalize=True`.
   """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

plot_confusion_matrix(confusion_matrix(y_true=y_test, y_pred=final_pred), classes=['0','1'])
# Print mean values stored in df_res df
print('Precision on the test set: ', round(df_res['test_precision'].mean(), 3))
print('Recall on the test set: ', round(df_res['test_recall'].mean(), 3))
print('F1 on the test set: ', round(df_res['test_f1'].mean(), 3))
fpr, tpr, thresholds = roc_curve(y_test, np.mean(predictions_proba, axis=0))
auc = roc_auc_score(y_test, np.mean(predictions_proba, axis=0))

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()
print('Area Under Curve: {:.3}'.format(auc))
append_results = False #overwrite results if append_result = False
filename = 'output'

output = '{}_{}.xlsx'.format(filename, model_name)
if append_results:
    if(os.path.exists(output)):
        print('* * * Writing file {}'.format(output))
        df_file = pd.read_excel(output)
        df_fin = pd.concat([df_file, df_res])
        df_fin.to_excel(output)   
    else:
        df_res.to_excel(output)
else:
    df_res.to_excel(output)