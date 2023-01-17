
# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
 
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
 
 
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
 
# %% [code]
#!!!!
#
# License: BSD 3 clause
 
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
import time
import warnings
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import time
 
 
 
 
def hyperparameter_tune(base_model, parameters, n_iter, kfold, X, y):
    start_time = time.time()
    
    # Arrange data into folds with approx equal proportion of classes within each fold
    k = StratifiedKFold(n_splits=kfold, shuffle=False)
    
    optimal_model = RandomizedSearchCV(base_model,
                            param_distributions=parameters,
                            n_iter=n_iter,
                            cv=k,
                            n_jobs=-1,
                            random_state=43)
    
    optimal_model.fit(X, y)
    stop_time = time.time()
 
    scores = cross_val_score(optimal_model, X, y, cv=k, scoring="accuracy")
    
    print("Elapsed Time:", time.strftime("%H:%M:%S", time.gmtime(stop_time - start_time)))
    print("====================")
    print("Cross Val Mean: {:.3f}, Cross Val Stdev: {:.3f}".format(scores.mean(), scores.std()))
    print("Best Score: {:.3f}".format(optimal_model.best_score_))
    print("Best Parameters: {}".format(optimal_model.best_params_))
    
    return optimal_model.best_params_, optimal_model.best_score_, optimal_model
 
 
base_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)
 
lots_of_parameters = {
    "max_depth": [3, 5, 10, 15],
    "n_estimators": [ 200,  400, 500],
    "max_features": [10,20,30],
    "criterion": ["gini", "entropy"],
    "bootstrap": [True, False],
    "min_samples_leaf": [1,2,4,6,8,15,50],
 
}
parameters = {
    "algorithm":["SAMME"],
    "base_estimator__criterion": ["gini", "entropy"],
    "learning_rate": [0.1,0.5,0.8,1,1.5,3],
    "n_estimators": [100,300,500],
    #"base_estimator__min_samples_split": [1,3,5,10],
    #"base_estimator__min_samples_leaf": [1,3,5,10]
    
} #"algorithm": ["SAMME.R","SAMME"],
 

 
!pip install pycm
from pycm import *
from pycm import ConfusionMatrix
import statistics 
import time
from sklearn import metrics
 
 

 
def multiClassStat(model, X_test, y_test, y_pred):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print('accuracy ', accuracy)
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    print("precision ", precision)
    y_prob = model.predict_proba(X_test)
    try:
        roc_auc = metrics.roc_auc_score(y_test, y_prob, average='macro', multi_class='ovr')
    except ValueError as inst:
        print(inst)
    print("roc_auc ", roc_auc)
    conf_mat = metrics.multilabel_confusion_matrix(y_test, y_pred)
    TPR = 0
    FPR = 0
    for i in conf_mat:
        TN = i[0][0]
        FP = i[0][1]
        FN = i[1][0]
        TP = i[1][1]
        print(TP)
        TPR += 0 if (TP + FN) == 0 else TP / (TP + FN)
        FPR += 0 if (FP + TN) == 0 else FP / (FP + TN)
    TPR /= len(conf_mat)
    FPR /= len(conf_mat)
    print('TPR ', TPR)
    print('FPR ', FPR)
    PR_curve = 0
    #for each class
    for i,cls in zip(range(len(model.classes_)),model.classes_):
        y_test_ = list(map(int, [num == cls for num in y_test]))
        print(i)
        print(cls)
        print(y_test_)
        precision_, recall_, thresholds = metrics.precision_recall_curve(y_test_, y_prob[:, i])
        print("pr_curve ", metrics.auc(recall_, precision_))
        PR_curve += metrics.auc(recall_, precision_)
    print(len(y_prob[0]))
    PR_curve /= len(y_prob[0])
    print(PR_curve)
    return accuracy, TPR, FPR, precision, roc_auc, PR_curve
 
def binaryStat(model, X_test, y_test, y_pred):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    print('precision , ', precision)
    y_prob = model.predict_proba(X_test)
    try:
        roc_auc = metrics.roc_auc_score(y_test, y_prob[:, 1])
    except ValueError as inst:
        print(inst)
        roc_auc = None
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    TN = conf_mat[0][0]
    FP = conf_mat[0][1]
    FN = conf_mat[1][0]
    TP = conf_mat[1][1]
    TPR = 0 if (TP + FN) == 0 else TP / (TP + FN)
    FPR = 0 if (FP + TN) == 0 else FP / (FP + TN)
    _precision, _recall, thresholds = metrics.precision_recall_curve(y_test, y_prob[:,1], )
    PR_curve = metrics.auc(_recall, _precision)
 
    return accuracy, TPR, FPR, precision, roc_auc, PR_curve
 
 
 
 
# %% [code]
 
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from scipy.stats import randint
from scipy.stats import uniform

model_params = {
    "algorithm":["SAMME"],
    "base_estimator__criterion": ["entropy"],
    "learning_rate": uniform(0.01,0.3),
    "n_estimators": randint(50,300),
    "base_estimator__min_samples_split": randint(1,20),
    #"base_estimator__min_samples_leaf": [1,5,30]
    }
 
 

# get datasets
 
dirname = "../input/classification-datasets/classification_datasets"

#create file for output

df_redults = pd.DataFrame(columns=['dataset_name', 'algorithm_name', 'cross_validation', 'hyper_params',
                          'accuracy', 'TPR', 'FPR', 'precision', 'roc_auc', 'PR_curve', 'training_time',
                          'inference_time'])
 
 
 
 
df_redults.to_csv('results.csv')
for filename in os.listdir(dirname): 
    
    print()
    print(filename)
    data = pd.read_csv(dirname+'/'+filename)
    print(data.dtypes)
    data_count_target = data[data.columns[-1]].value_counts()
    # remove classes with less than 10 lines
    cls_count=0
    dropped=False
    for cls, cnt in data_count_target.iteritems():
        cls_count += 1
        if cnt < 10:
            data = data[data[data.columns[-1]] != cls]
            cls_count -= 1
            dropped=True
    if cls_count < 2:
        print("dropping file:  ", filename)
        continue
    # convert to 0 1 labels
    if cls_count==2 and dropped:
        max_val = data[data.columns[-1]].value_counts().index[0]
        data[data.columns[-1]] = data[data.columns[-1]].apply(lambda x: 0 if x == max_val else 1)
    # strings- convert using  LabelEncoder
    for i in data.columns:
        if data[i].dtype == np.string_:
            is_string_type=True
        else:
            is_string_type=False
        
        if is_string_type:
            print(i)
            enc = LabelEncoder()
            data[i] = enc.fit_transform(data[i].astype(str))
    data.dropna(inplace=True)
    print(data)
    
    le = preprocessing.LabelEncoder()
    for column in data.columns:
        data[column]=le.fit_transform(data[column])
    
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
  
    #####
    #####
    ######
 
 
    #label encoder from categorial to int categories
 
    for column_name in X.columns:
        if X[column_name].dtype == object:
            X[column_name] = le.fit_transform(X[column_name])
        else:
            pass
 
 
    
    skf = StratifiedKFold(n_splits=10)
    fold=0
    
    for train_index, test_index in skf.split(X, y):
        fold+=1
        print('fold: ',fold)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        bdt_real_ = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=10,learning_rate=0.5)
        best_params,result,optimal_model = hyperparameter_tune(bdt_real_, model_params, 50, 3, X_train, y_train)
 
        
 
        #optimal_model = RandomizedSearchCV(bdt_real_,
        #                param_distributions=model_params,
        #                n_iter=50,
        #                cv=3,
        #                n_jobs=-1,
        #                random_state=43)
        
        #bdt_real=optimal_model
        
        #** 
        bdt_real=optimal_model
        start = time.time()
        bdt_real.fit(X_train, y_train)
        training_time = time.time() - start
        start = time.time()
        y_pred = bdt_real.predict(X_test)
        inference_time = (time.time() - start) / len(X_test) # single inference time
        inference_time *= 1000 # 1000 lines
        
        #######
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        if len(y_test.unique()) == 2:
            
            accuracy, TPR, FPR, precision, roc_auc, PR_curve = binaryStat(bdt_real, X_test, y_test, y_pred)
        else:
            accuracy, TPR, FPR, precision, roc_auc, PR_curve = multiClassStat(bdt_real, X_test, y_test, y_pred)
        df_redults = df_redults.append({'dataset_name': filename , 'algorithm_name':'Multi-Class-AdaBoostClassifier',
                                        'cross_validation': fold, 'hyper_params': optimal_model.best_params_,
                                        'accuracy': accuracy, 'TPR': TPR, 'FPR': FPR, 'precision': precision,
                                        'roc_auc':roc_auc, 'PR_curve':PR_curve, 'training_time':training_time,
                                        'inference_time':inference_time}, ignore_index=True)
        print( optimal_model.best_params_, accuracy, TPR, FPR, precision, roc_auc, PR_curve,
              training_time, inference_time)
    df_redults.to_csv('results.csv', mode='a', header=False)
print ("finish")
