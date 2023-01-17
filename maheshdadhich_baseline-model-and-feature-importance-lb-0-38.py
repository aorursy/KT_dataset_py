import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import time
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import warnings
from sklearn import preprocessing
import xgboost as xgb
import pickle
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import itertools
%matplotlib inline
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# sanity check 
print("Shape of train and test are {} and {} respectively".format(train.shape, test.shape))
print("Number of nulls in train and test are {} and {} respectively".format(train.isnull().sum().sum(), test.isnull().sum().sum()))
# we actually won't worry as XGB handles missing values well.
# distribution of target
start = time.time()
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)
sns.despine(left=True)
sns.distplot(train.Target.values, axlabel = "target_values", label = 'frequency', color="r", kde = False)
plt.setp(axes, yticks=[])
plt.tight_layout()
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))
plt.show()
# Checking columns which are not numeric and convertiong them to factors 
print("Number of columns in train file is {}".format(train.shape[1]))
# checking columns that has values other than numbers 
non_numeric_cols = []
for col in train.columns:
    if np.issubdtype(train[col].dtype, np.number) == False:
        non_numeric_cols.append(col)
print("Number of non numeric columns {}".format(non_numeric_cols.__len__()))
print(non_numeric_cols)  

# we has to remove id and has to convert all the other variables into factors 
cols_to_factor = ['idhogar', 'dependency', 'edjefe', 'edjefa']
def Label_for_cat_var(df, col):
    """Function to define labels for categorical columns"""
    le = preprocessing.LabelEncoder()
    le.fit(df[col])
    df[col] = le.transform(df[col])
    del le
    return(df)

t0 = time.time()
for col in cols_to_factor:
    train_num = Label_for_cat_var(train, col)
t1 = time.time()
print("Taken taken in converting categorical variables to numberic in train is "+ str(t1-t0))

# for test set 
t0 = time.time()
for col in cols_to_factor:
    test_num = Label_for_cat_var(test, col)
t1 = time.time()
print("Taken taken in converting categorical variables to numberic in test is "+ str(t1-t0))

def train_best_model(target_stats, train, target_var_xgboost, features_name):
    """Function to train and save best model for given target variable list
        Input -
                target_stats - a dict performance for each target files
                train - train dataframe contaning all the variables that are needed for xgb
                target_var_xgboost - List containing all the target variables for xgb
        Output - 
    """
    for target in target_var_xgboost:
        index_max = target_stats[target]['score'].index(max(target_stats[target]['score']))
        parameter_for_max = target_stats[target]['parameters'][index_max]
        y = train[target].values
        Xtr, Xv, ytr, yv = train_test_split(train[features_name].values, y, test_size=0.2, random_state=1987)
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        dvalid = xgb.DMatrix(Xv, label=yv)
        #dtest = xgb.DMatrix(test[temp].values)
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        xgb_par = parameter_for_max
        model = xgb.train(xgb_par, dtrain, 2000, watchlist, early_stopping_rounds=50,
                          maximize=False, verbose_eval=100)
        
        return(model)
train_, test_ = train_test_split(train_num, test_size=0.2, random_state=1987)
train_.shape
target_var_xgboost = ['Target']
cols = train_num.columns
features_name = [x for x in cols if x !='Id']
features_name = [x for x in features_name if x !='Target']
print("number of features {}".format(features_name.__len__()))
target_stats = {'Target': {'parameters': [{'booster': 'gbtree',
                'colsample_bytree': 0.3,
                'eta': 0.1,
                'eval_metric': 'mlogloss',
                'lambda': 2.0,
                'max_depth': 6,
                'min_child_weight': 10,
                'nthread': -1,
                'num_class': 15,
                'objective': 'multi:softmax',
                'silent': 1,
                'subsample': 0.9}],
              'score': [0.362916]}}
model = train_best_model(target_stats, train_, target_var_xgboost, features_name)
def variable_importance(model, features_name):
    """Function to calculate the variable importance for model
        Input - 
                model - xgb model 
        Output -
                var_imp_dict - dict of variable importance 
    """
    feature_importance_dict = model.get_fscore()
    fs = ['f%i' % i for i in range(len(features_name))]
    new_feat_number = [x[1:] for x in feature_importance_dict.keys()]
    
    f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()),
                       'feature_names': [features_name[int(x)] for x in new_feat_number],
                       'importance': list(feature_importance_dict.values())}).sort_values(by='importance', ascending=False)


    return(f1)



feature_importance = variable_importance(model, features_name)
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(17, 7), sharex=True)
sns.despine(left=True)
sns.barplot(x = feature_importance['feature_names'].head(10), y =feature_importance['importance'].head(10) ) 
plt.show()
# Checking confusion matrix on validation datasets 
# predictions and checking the performance 
start = time.time()
dtest = xgb.DMatrix(test_[features_name].values)
#yvalid = model.predict(dvalid)
ytest = model.predict(dtest)
end = time.time()
print("Time taken in prediction is {}.".format(end - start))
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
    fig = plt.figure(figsize = (11,11))
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
    plt.show()
ygt = test_.Target.values
label = list(set(ygt))
cnf_matrix = confusion_matrix(ygt, ytest, label)


plot_confusion_matrix(cnf_matrix, label,
                      normalize=False,
                      title='Confusion matrix',
                      cmap=plt.cm.Blues)
# predictions and checking the performance 
start = time.time()
dtest = xgb.DMatrix(test_num[features_name].values)
#yvalid = model.predict(dvalid)
ytest = model.predict(dtest)
end = time.time()
print("Time taken in prediction is {}.".format(end - start))
ytest = model.predict(dtest)
print('Test shape OK.') if test_num.shape[0] == ytest.shape[0] else print('Oops')
test_num['Target'] = ytest
test_num['Target'] = list(map(int, test_num['Target']))
test_num[['Id', 'Target']].to_csv('BuryBury_xgb_submission.csv', index=False)