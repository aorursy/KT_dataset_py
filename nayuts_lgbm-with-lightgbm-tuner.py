import gc

from itertools import cycle

import random



import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import interp

import seaborn as sns

from sklearn.metrics import f1_score

from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split



from optuna.integration import lightgbm as lgb

#import lightgbm as lgb
def fix_seed(seed):

    # random

    random.seed(seed)

    # Numpy

    np.random.seed(seed)



SEED = 42

fix_seed(SEED)
!ls ../input/fetal-health-classification
fetal_health = pd.read_csv("../input/fetal-health-classification/fetal_health.csv")
fetal_health.head()
fetal_health.info()
fetal_health.describe()
def plot_with_seaborn(fetal_health):

    fig, axes = plt.subplots(11, 2, figsize=(10,40))

    fig.suptitle(f"Distributions of values of dataset")

    g1 = sns.distplot(fetal_health["baseline value"],  color='orange', ax=axes[0, 0])

    g2 = sns.distplot(fetal_health["accelerations"], color='darkgoldenrod', ax=axes[0, 1])

    g3 = sns.distplot(fetal_health["fetal_movement"], color='darkkhaki', ax=axes[1, 0])

    g4 = sns.distplot(fetal_health["uterine_contractions"], color='olive', ax=axes[1, 1])

    g5 = sns.distplot(fetal_health["light_decelerations"], color='lime', ax=axes[2, 0])

    g6 = sns.countplot(fetal_health["severe_decelerations"], ax=axes[2, 1])

    g7 = sns.countplot(fetal_health["prolongued_decelerations"], ax=axes[3, 0])

    g8 = sns.distplot(fetal_health["abnormal_short_term_variability"], color='blue', ax=axes[3, 1])

    g9 = sns.distplot(fetal_health["mean_value_of_short_term_variability"], color='violet', ax=axes[4, 0])

    g10 = sns.distplot(fetal_health["percentage_of_time_with_abnormal_long_term_variability"], color='darkmagenta', ax=axes[4, 1])

    g11 = sns.distplot(fetal_health["mean_value_of_long_term_variability"], color='orange', ax=axes[5, 0])

    g12 = sns.distplot(fetal_health["histogram_width"], color='darkgoldenrod', ax=axes[5, 1])

    g13 = sns.distplot(fetal_health["histogram_min"], color='darkkhaki', ax=axes[6, 0])

    g14 = sns.distplot(fetal_health["histogram_max"], color='olive', ax=axes[6, 1])

    g15 = sns.distplot(fetal_health["histogram_number_of_peaks"], color='lime', ax=axes[7, 0])

    g16 = sns.countplot(fetal_health["histogram_number_of_zeroes"], ax=axes[7, 1])

    g17 = sns.distplot(fetal_health["histogram_mode"], color='darkturquoise', ax=axes[8, 0])

    g18 = sns.distplot(fetal_health["histogram_mean"], color='blue', ax=axes[8, 1])

    g19 = sns.distplot(fetal_health["histogram_median"], color='violet', ax=axes[9, 0])

    g20 = sns.distplot(fetal_health["histogram_variance"], color='darkmagenta', ax=axes[9, 1])

    g21 = sns.distplot(fetal_health["histogram_tendency"], color='orange', ax=axes[10, 0])

    g22 = sns.countplot(fetal_health["fetal_health"], ax=axes[10, 1])
plot_with_seaborn(fetal_health)
fig = plt.figure(figsize=(15, 15))

corr = fetal_health.corr()

sns.heatmap(corr, square=True, annot=True)
fig = plt.figure(figsize=(15, 15))

sns.pairplot(fetal_health[corr[abs(corr["fetal_health"]) > 0.30].index])
fig = plt.figure(figsize=(15, 15))

sns.pairplot(fetal_health[corr[abs(corr["fetal_health"]) <= 0.30].index])
X = fetal_health[[col for col in fetal_health.columns if col not in ["fetal_health"]]]

y = fetal_health["fetal_health"]
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)

X = X.reset_index(drop=True)

y = y.reset_index(drop=True)

X_test = X_test.reset_index(drop=True) 

y_test = y_test.reset_index(drop=True)



#To use LightGBM's multiclass objective, I adjust labels.

y = y - 1

y_test = y_test - 1
params = {

    "objective": "multiclass",

    "boosting": "gbdt",

    "num_leaves": 40,

    "learning_rate": 0.05,

    "feature_fraction": 0.85,

    "reg_lambda": 2,

    "metric": "multi_logloss",

    "num_class" : 3,

}
def calc_multiclass_auc(y_test, y_pred):

    y_test = label_binarize(y_test, classes=[0, 1, 2])

    y_pred = label_binarize(y_pred, classes=[0, 1, 2])

    

    fpr = dict()

    tpr = dict()

    roc_auc = dict()

    for i in range(3):

        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])

        roc_auc[i] = auc(fpr[i], tpr[i])



    # Compute micro-average ROC curve and ROC area

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())

    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    

    return roc_auc, tpr, fpr

    
kf = KFold(n_splits=3)

models = []

f1 = 0

auc_vals = []

tpr_vals = []

fpr_vals = []



for train_index,val_index in kf.split(X):

    train_features = X.loc[train_index]

    train_target = y.loc[train_index]

    

    val_features = X.loc[val_index]

    val_target = y.loc[val_index]

    

    d_training = lgb.Dataset(train_features, label=train_target, free_raw_data=False)

    d_val = lgb.Dataset(val_features, label=val_target, free_raw_data=False)

    

    cls = lgb.train(params, train_set=d_training, num_boost_round=1000, valid_sets=[d_val], verbose_eval=25, early_stopping_rounds=50)



    models.append(cls)

    f1 += f1_score(val_target, np.argmax(cls.predict(val_features),axis=1), average='macro')

    

    roc_auc, tpr, fpr = calc_multiclass_auc(val_target, np.argmax(cls.predict(val_features),axis=1))

    auc_vals.append(roc_auc)

    tpr_vals.append(tpr)

    fpr_vals.append(fpr)

def print_tuned_params(cls, fold):

    print("---------------------")

    print(f"Tune result of the {fold}th fold.")

    print("params:", cls.params)

    print("best_iteration:", cls.best_iteration)

    print("best_score:", cls.best_score)    

    print("---------------------")
fold = 1

for cls in models:

    print_tuned_params(cls, fold)

    fold += 1



print("F1 score:", f1 / 3)
auc_cal_micro = 0

auc_val_0 = 0

auc_val_1 = 0

auc_val_2 = 0



for auc_val in auc_vals:

    auc_cal_micro += auc_val['micro']

    auc_val_0 += auc_val[0]

    auc_val_1 += auc_val[1]

    auc_val_2 += auc_val[2]

    

print("auc micro", auc_cal_micro / 3)

print("auc for label 0:", auc_val_0 / 3)

print("auc for label 1:", auc_val_1 / 3)

print("auc for label 2:", auc_val_2 / 3)
def plot_roc_curve(tprs, fprs, fold,  n_classes, lw):

    """Plots ROC curves for the multilabel problem

    Refer https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    """

    if fold != "test":

        tpr = tprs[fold]

        fpr = fprs[fold]

    else:

        tpr = tprs

        fpr = fprs

        

    n_classes = 3

    lw = 2

    

    # First aggregate all false positive rates

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



    # Then interpolate all ROC curves at this points

    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):

        mean_tpr += interp(all_fpr, fpr[i], tpr[i])



    # Finally average it and compute AUC

    mean_tpr /= n_classes



    fpr["macro"] = all_fpr

    tpr["macro"] = mean_tpr

    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



    # Plot all ROC curves

    plt.figure()

    plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='deeppink', linestyle=':', linewidth=4)



    plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=4)



    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

    for i, color in zip(range(n_classes), colors):

        plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))



    plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title(f'Roc Curve of fold {fold}')

    plt.legend(loc="lower right")

    plt.show()
plot_roc_curve(tpr_vals, fpr_vals, 0,  3, 2)
plot_roc_curve(tpr_vals, fpr_vals, 1,  3, 2)
plot_roc_curve(tpr_vals, fpr_vals, 2,  3, 2)
result = np.zeros((X_test.shape[0], 3))
for model in models:

    result += model.predict(X_test)

f1_test = f1_score(y_test, np.argmax(result,axis=1), average='macro')

auc_test, tpr_test, fpr_test = calc_multiclass_auc(y_test, np.argmax(result,axis=1))
f1_test
plot_roc_curve(tpr_test, fpr_test, "test",  3, 2)
print("Predicted fetal_health labels are:")

np.argmax(result,axis=1) + 1