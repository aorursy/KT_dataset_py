from collections import Counter

import category_encoders as ce

from datetime import timedelta 

from datetime import datetime

from scipy import interp

import pandas as pd

import numpy as np

import itertools

import warnings



%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams



from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, auc

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
def read_data(file_path):

    print('Loading datasets...')

    train = pd.read_csv(file_path + 'train.csv', sep=',')

    test = pd.read_csv(file_path + 'test.csv', sep=',')

    print('Datasets loaded')

    return train, test
PATH = '../input/cat-in-the-dat-ii/'

train, test = read_data(PATH)
### Zoom datasets 

def zoom_dataset(data):

    Count_missing_val = data.isnull().sum()

    Percent_missing = (data.isnull().sum()/data.isnull().count()*100)

    Percent_no_missing = 100 - Percent_missing

    Count_unique = data.nunique()

    Percent_unique_val = Count_unique / len(data)*100

    Type = data.dtypes

    data=[[i, Counter(data[i][data[i].notna()]).most_common(3)] for i in list(data)]

    top = pd.DataFrame(data, columns=['Name', 'Top_3']).set_index(['Name'])

    tt = pd.concat([Percent_no_missing, Count_missing_val, Percent_missing, Count_unique, Percent_unique_val, Type], axis=1, 

                   keys=['Percent_no_missing','Count_missing_val', 'Percent_missing', ' Count_unique', 'Percent_unique_val', 'Type'])

    tt = pd.concat([tt, top[['Top_3']]], axis=1, sort=False).reset_index()

    return((tt))
#overview train - test

for i in [train, test]:

    display(zoom_dataset(i))
# train

f,ax=plt.subplots(8,2,figsize=(12,20))

f.delaxes(ax[7,1])



for i,feature in enumerate(['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4']):

    colors = ['darkturquoise', 'darkorange']

    sns.countplot(x=feature,data=train,hue='target',ax=ax[i//2,i%2], palette = colors, alpha=0.7, edgecolor=('black'), linewidth=2)

    ax[i//2,i%2].grid(b=True, which='major', color='grey', linewidth=0.4)

    ax[i//2,i%2].set_title('Count of {} vs target - train'.format(feature), fontsize=18)

    ax[i//2,i%2].legend(loc='best')

    ax[i//2,i%2].set_ylabel('count', fontsize=12)

    ax[i//2,i%2].set_xlabel('modality', fontsize=12)



plt.tight_layout()

plt.show()
# test

f,ax=plt.subplots(8,2,figsize=(12,20))

f.delaxes(ax[7,1])



for i,feature in enumerate(['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4']):

    colors = ['darkturquoise']

    sns.countplot(x=feature,data=test,ax=ax[i//2,i%2], palette = colors, alpha=0.8, edgecolor=('black'), linewidth=2)

    ax[i//2,i%2].grid(b=True, which='major', color='grey', linewidth=0.2)

    ax[i//2,i%2].set_title('Count of {} - test'.format(feature), fontsize=18)

    ax[i//2,i%2].set_ylabel('count', fontsize=12)

    ax[i//2,i%2].set_xlabel('modality', fontsize=12)



plt.tight_layout()

plt.show()
# CREDITS : https://www.kaggle.com/caesarlupum/2020-20-lines-target-encoding

def encoding(train, test, smooth):

    print('Target encoding...')

    train.sort_index(inplace=True)

    target = train['target']

    test_id = test['id']

    train.drop(['target', 'id'], axis=1, inplace=True)

    test.drop('id', axis=1, inplace=True)

    cat_feat_to_encode = train.columns.tolist()

    smoothing=smooth

    oof = pd.DataFrame([])

    for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=2020, shuffle=True).split(train, target):

        ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

        ce_target_encoder.fit(train.iloc[tr_idx, :], target.iloc[tr_idx])

        oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)

    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

    ce_target_encoder.fit(train, target)

    train = oof.sort_index()

    test = ce_target_encoder.transform(test)

    features = list(train)

    print('Target encoding done!')

    return train, test, test_id, features, target
# Timer

def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('Time taken : %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
# Confusion matrix

def plot_confusion_matrix(cm, classes,

                          normalize = False,

                          title = 'Confusion matrix',

                          cmap = plt.cm.Blues) :

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)

    plt.title(title, fontsize=12)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation = 0)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :

        plt.text(j, i, cm[i, j],

                 horizontalalignment = 'center',

                 color = 'white' if cm[i, j] > thresh else 'black')

 

    plt.tight_layout()

    plt.ylabel('True label', fontsize=12)

    plt.xlabel('Predicted label', fontsize=12)
# model

def run_model(splits, features, target, train):

    # paramaters / hyperparameters

    model = lgb.LGBMClassifier(**{

                    'learning_rate': 0.05,

                    'feature_fraction': 0.1,

                    'min_data_in_leaf' : 12,

                    'max_depth': 3,

                    'reg_alpha': 1,

                    'reg_lambda': 1,

                    'objective': 'binary',

                    'metric': 'auc',

                    'n_jobs': -1,

                    'n_estimators' : 5000,

                    'feature_fraction_seed': 42,

                    'bagging_seed': 42,

                    'boosting_type': 'gbdt',

                    'verbose': 1,

                    'is_unbalance': True,

                    'boost_from_average': False})

    

    # Graph size

    plt.rcParams['figure.figsize']=(6,4)

    

    start_time = timer(None)



    print('LGBM modeling...')

    

    # Metrics / fold

    cms= []

    tprs = []

    aucs = []

    y_real = []

    y_proba = []

    recalls = []

    roc_aucs = []

    mean_tpr = []

    mean_fpr = []

    f1_scores = []

    accuracies = []

    precisions = []



    oof = np.zeros(len(train))

    mean_fpr = np.linspace(0,1,100)

    feature_importance_df = pd.DataFrame()

    i = 1

    

    # Statified K Fold + model

    folds = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):

        print('Fold:', fold_ )

        model = model.fit(train.iloc[trn_idx][features], target.iloc[trn_idx],

                          eval_set = [(train.iloc[trn_idx][features], target.iloc[trn_idx]), 

                                      (train.iloc[val_idx][features], target.iloc[val_idx])],

                          verbose = 1000,

                          eval_metric = 'auc',

                          early_stopping_rounds = 1000)

        

        # oof

        oof[val_idx] =  model.predict_proba(train.iloc[val_idx][features])[:,1]

        

        # Roc curve / fold

        f = plt.figure(1)

        fpr, tpr, t = roc_curve(target[val_idx], oof[val_idx])

        tprs.append(interp(mean_fpr, fpr, tpr))

        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)

        plt.plot(fpr, tpr, lw=2, alpha=0.5, label='ROC fold %d (AUC = %0.4f)' % (i,roc_auc))



        # Precion recall / fold

        g = plt.figure(2)

        precision, recall, _ = precision_recall_curve(target[val_idx], oof[val_idx])

        y_real.append(target[val_idx])

        y_proba.append(oof[val_idx])

        plt.plot(recall, precision, lw=2, alpha=0.3, label='P|R fold %d' % (i))  



        i= i+1



        # append metric by fold

        roc_aucs.append(roc_auc_score(target[val_idx], oof[val_idx]))

        accuracies.append(accuracy_score(target[val_idx], oof[val_idx].round()))

        recalls.append(recall_score(target[val_idx], oof[val_idx].round()))

        precisions.append(precision_score(target[val_idx], oof[val_idx].round()))

        f1_scores.append(f1_score(target[val_idx], oof[val_idx].round()))



        # Confusion matrix 

        cms.append(confusion_matrix(target[val_idx], oof[val_idx].round()))



    # Means

    print(

        '\nCV roc score        : {0:.4f}, std: {1:.4f}.'.format(np.mean(roc_aucs), np.std(roc_aucs)),

        '\nCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),

        '\nCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),

        '\nCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),

        '\nCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores)))

    

    print('Modeling done!\n')

    

    # Timer end

    timer(start_time)

    

    return model, cms, tprs, aucs, y_real, y_proba, mean_fpr, mean_tpr, train



def graph_metrics(cms, tprs, aucs, y_real, y_proba, mean_fpr, mean_tpr):



    # Roc curve

    f = plt.figure(1)

    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')

    mean_tpr = np.mean(tprs, axis=0)

    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(mean_fpr, mean_tpr, color='blue',

             label=r'Mean ROC (AUC = %0.4f)' % ((mean_auc)),lw=2, alpha=1)

    plt.grid(b=True, which='major', color='grey', linewidth=0.4)

    plt.xlabel('False Positive Rate', fontsize=12)

    plt.ylabel('True Positive Rate', fontsize=12)

    plt.title('LGBM - ROC by folds', fontsize=18)

    plt.legend(loc="lower right")



    # Recall precision curve

    g = plt.figure(2)

    plt.plot([0,1],[1,0],linestyle = '--',lw = 2,color = 'grey')

    y_real = np.concatenate(y_real)

    y_proba = np.concatenate(y_proba)

    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    plt.plot(recall, precision, color='blue',

             label=r'Mean P|R')

    plt.grid(b=True, which='major', color='grey', linewidth=0.4)

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.title('LGBM P|R curve by folds', fontsize=18)

    plt.legend(loc="lower left")



    # Confusion matrix

    plt.rcParams["axes.grid"] = False

    cm = np.average(cms, axis=0)

    class_names = [0,1]

    plt.figure()

    plot_confusion_matrix(cm, 

                          classes=class_names, 

                          title='LGBM Confusion matrix [averaged/folds]')

    plt.show()
# Encoding

train, test, test_id, features, target = encoding(train, test, 0.3)

# Modeling

model, cms, tprs, aucs, y_real, y_proba, mean_fpr, mean_tpr, df_ml = run_model(5, features, target, train)

# Metrics + dataviz

graph_metrics(cms, tprs, aucs, y_real, y_proba, mean_fpr, mean_tpr)
pd.DataFrame({'id': test_id, 'target': model.predict_proba(test)[:,1]}).to_csv('submission.csv', index=False)