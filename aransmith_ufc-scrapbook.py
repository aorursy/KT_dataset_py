# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



def normalize(df):

    df_num = df.select_dtypes(include=[np.float, np.int])

    scaler = StandardScaler()

    df[list(df_num.columns)] = scaler.fit_transform(df[list(df_num.columns)])



# get data 

ufc_pp_data = pd.read_csv("/kaggle/input/ufcdata/preprocessed_data.csv")



# normalize all except one-hots, non-ints, non-floats

# backup weight_class_* , B_Stance_*, R_Stance_*

ufc_pp_one_hots = ufc_pp_data.filter(regex=("weight_class_*|R_Stance_*|B_Stance_*"))



# disclude columns above

ufc_pp_no_hots = ufc_pp_data.drop(ufc_pp_one_hots.columns, axis=1)



# normalize

normalize(ufc_pp_no_hots)



# recombine the one-hots

ufc_pp_normalized = pd.concat([ufc_pp_no_hots, ufc_pp_one_hots], axis=1)



# seperate the data

y = ufc_pp_normalized['Winner']

X = ufc_pp_normalized.drop(columns = 'Winner')



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=43)



# fit model

model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=43)

model.fit(X_train, y_train)



y_preds = model.predict(X_test)



# test model

print("OOB, Accuracy\n", model.oob_score_, accuracy_score(y_test, y_preds))
import category_encoders as ce



# get data, going to need the original data to revert the one hot encoding

ufc_raw_data = pd.read_csv("/kaggle/input/ufcdata/data.csv")



# normalize all except one-hots, non-ints, non-floats

# backup weight_class_* , B_Stance_*, R_Stance_*

ufc_pp_one_hots = ufc_pp_data.filter(regex=("weight_class_*|R_Stance_*|B_Stance_*"))



# disclude columns above

ufc_pp_no_hots = ufc_pp_data.drop(ufc_pp_one_hots.columns, axis=1)



# normalize

normalize(ufc_pp_no_hots)

        

# reverse the stance column

ufc_rev_one_hots = pd.DataFrame({})



ufc_pp_weight_class = ufc_pp_data.filter(regex=("weight_class_*"))

ufc_rev_one_hots["weight_class"] = ufc_pp_weight_class.idxmax(1)



ufc_pp_R_Stance = ufc_pp_data.filter(regex=("R_Stance_*"))

ufc_rev_one_hots["R_Stance"] = ufc_pp_R_Stance.idxmax(1)



ufc_pp_B_Stance = ufc_pp_data.filter(regex=("B_Stance_*"))

ufc_rev_one_hots["B_Stance"] = ufc_pp_B_Stance.idxmax(1)



# binary encode the categories

encoder = ce.BinaryEncoder(cols=ufc_rev_one_hots.columns.tolist())

ufc_cat_binary = encoder.fit_transform(ufc_rev_one_hots)



# combine the binary categories with data

ufc_pp_normalized = pd.concat([ufc_pp_no_hots, ufc_cat_binary], axis=1)



# seperate the data

y = ufc_pp_normalized['Winner']

X = ufc_pp_normalized.drop(columns = 'Winner')



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=43)



# fit model

model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=43)

model.fit(X_train, y_train)



y_preds = model.predict(X_test)



# test model

print("OOB, Accuracy\n", model.oob_score_, accuracy_score(y_test, y_preds))
import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt



mpl.style.use('seaborn')



# backup weight_class_* , B_Stance_*, R_Stance_*

ufc_pp_one_hots = ufc_pp_data.filter(regex=("weight_class_*|R_Stance_*|B_Stance_*"))



# disclude columns above

ufc_pp_no_hots = ufc_pp_data.drop(ufc_pp_one_hots.columns, axis=1)



# reverse the stance column

ufc_rev_one_hots = pd.DataFrame({})



ufc_pp_weight_class = ufc_pp_data.filter(regex=("weight_class_*"))

ufc_rev_one_hots["weight_class"] = ufc_pp_weight_class.idxmax(1)



ufc_pp_R_Stance = ufc_pp_data.filter(regex=("R_Stance_*"))

ufc_rev_one_hots["R_Stance"] = ufc_pp_R_Stance.idxmax(1)



ufc_pp_B_Stance = ufc_pp_data.filter(regex=("B_Stance_*"))

ufc_rev_one_hots["B_Stance"] = ufc_pp_B_Stance.idxmax(1)



# combine the binary categories with data

ufc_pp_no_hots = pd.concat([ufc_pp_no_hots, ufc_rev_one_hots], axis=1)



ufc_pp_no_hots.info(verbose=True)
# ax = sns.distplot(ufc_pp_data.Winner)

ax = sns.countplot(ufc_pp_no_hots.Winner)

plt.show()



value_counts = ufc_pp_no_hots.Winner.value_counts()

print(value_counts / (value_counts[0] + value_counts[1]))



ax = sns.countplot(ufc_pp_no_hots.B_Stance)

plt.show()

ax = sns.countplot(ufc_pp_no_hots.R_Stance)

plt.show()
ax = sns.distplot(ufc_pp_no_hots.R_Reach_cms)

plt.show()

ax = sns.distplot(ufc_pp_no_hots.B_Reach_cms)

plt.show()
reach_diff = ufc_pp_no_hots.R_Reach_cms - ufc_pp_no_hots.B_Reach_cms



#winner_reach_diff.sort_values('')

ax = sns.distplot(reach_diff)

plt.show()



# reach_diff = reach_diff.to_frame()

# reach_diff = reach_diff.rename(columns= {0: 'reach_diff'})

winner_reach_diff = pd.concat([reach_diff, ufc_pp_no_hots.Winner], axis=1)

winner_reach_diff = winner_reach_diff.rename(columns= {0: 'reach_diff'})

winner_reach_diff = winner_reach_diff.sort_values('reach_diff')
# what effect does reach difference have? 

sns.catplot(x="reach_diff", y="Winner", kind="box", data=winner_reach_diff);
from sklearn import datasets

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from hyperopt import tpe, hp, fmin

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_curve, auc

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt



# normalize generic dataframes inplace

# df: pandas dataframe

def normalize(df):

    df_num = df.select_dtypes(include=[np.float, np.int])

    scaler = StandardScaler()

    df[list(df_num.columns)] = scaler.fit_transform(df[list(df_num.columns)])



# from Yacin Nouri, https://www.kaggle.com/ynouri/random-forest-k-fold-cross-validation

def compute_roc_auc(index, clf):

    y_predict = clf.predict_proba(X.iloc[index])[:,1]

    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)

    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score    



# load the data

ufc_pp_data = pd.read_csv("/kaggle/input/ufcdata/preprocessed_data.csv")

normalize(ufc_pp_data)

criterions = ['gini', 'entropy']



y = ufc_pp_data['Winner'].astype('category').cat.codes

X = ufc_pp_data.drop(columns = 'Winner')



def objective_func(args):

    # return the average of the roc performance

    

    clf = RandomForestClassifier(n_estimators=args['params']['estimators'],

    criterion=args['params']['criterion'],

    max_depth=args['params']['max_depth'],

    min_samples_split=args['params']['min_samples_split'],

    min_samples_leaf=args['params']['min_samples_leaf'],

    min_weight_fraction_leaf=0,

    max_features='auto',

    max_leaf_nodes=None,

    min_impurity_decrease=0.0,

    min_impurity_split=None,

    bootstrap=True,

    oob_score=False,

    n_jobs=-1,

    random_state=43,

    verbose=0,

    warm_start=False,

    class_weight='balanced')



    cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)

    results = pd.DataFrame(columns=['training_score', 'test_score'])

    fprs, tprs, scores = [], [], []

        

    for (train, test), i in zip(cv.split(X, y), range(5)):

        clf.fit(X.iloc[train], y.iloc[train])

        _, _, auc_score_train = compute_roc_auc(train, clf)

        fpr, tpr, auc_score = compute_roc_auc(test, clf)

        scores.append((auc_score_train, auc_score))

        fprs.append(fpr)

        tprs.append(tpr)

    

    # return the average performance on the test splits

    total = 0

    for i in scores:

        total += i[1]

        

    # minimize average inaccuracy across 5 folds

    return 1 - total / len(scores)



space = {'params': {'estimators': hp.choice('estimators',range(50,250)),

        'criterion':hp.choice('criterion',criterions),

        'max_depth':hp.choice('max_depth',range(2,10)),

        'min_samples_split':hp.uniform('min_samples_split',0,0.99),

       'min_samples_leaf':hp.uniform('min_samples_leaf',0,0.49)}

        }



best_classifier = fmin(objective_func,space,

                        algo=tpe.suggest,max_evals=500)



print(best_classifier)
# this is one result that was found

best_classifier = {'criterion': 0, 'estimators': 71, 'max_depth': 6, 'min_samples_leaf': 2.0576639506899924e-05, 'min_samples_split': 0.0007085464011934425}
# from Yacin Nouri, https://www.kaggle.com/ynouri/random-forest-k-fold-cross-validation

def plot_roc_curve(fprs, tprs):

    """Plot the Receiver Operating Characteristic from a list

    of true positive rates and false positive rates."""

    

    # Initialize useful lists + the plot axes.

    tprs_interp = []

    aucs = []

    mean_fpr = np.linspace(0, 1, 100)

    f, ax = plt.subplots(figsize=(14,10))

    

    # Plot ROC for each K-Fold + compute AUC scores.

    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):

        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))

        tprs_interp[-1][0] = 0.0

        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)

        ax.plot(fpr, tpr, lw=1, alpha=0.3,

                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        

    # Plot the luck line.

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',

             label='Luck', alpha=.8)



    # Plot the mean ROC.

    mean_tpr = np.mean(tprs_interp, axis=0)

    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)

    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='b',

             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),

             lw=2, alpha=.8)

    

    # Plot the standard deviation around the mean ROC.

    std_tpr = np.std(tprs_interp, axis=0)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,

                     label=r'$\pm$ 1 std. dev.')

    

    # Fine tune and show the plot.

    ax.set_xlim([-0.05, 1.05])

    ax.set_ylim([-0.05, 1.05])

    ax.set_xlabel('False Positive Rate')

    ax.set_ylabel('True Positive Rate')

    ax.set_title('Receiver operating characteristic')

    ax.legend(loc="lower right")

    plt.show()

    return (f, ax)



# construct model with parameters

clf = RandomForestClassifier(n_estimators=best_classifier['estimators'],

    criterion=criterions[best_classifier['criterion']],

    max_depth=best_classifier['max_depth'],

    min_samples_split=best_classifier['min_samples_split'],

    min_samples_leaf=best_classifier['min_samples_leaf'],

    min_weight_fraction_leaf=0,

    max_features='auto',

    max_leaf_nodes=None,

    min_impurity_decrease=0.0,

    min_impurity_split=None,

    bootstrap=True,

    oob_score=False,

    n_jobs=-1,

    random_state=43,

    verbose=0,

    warm_start=False,

    class_weight='balanced')



# with the best parameters found, let's visualize its AUC curve

cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)

results = pd.DataFrame(columns=['training_score', 'test_score'])

fprs, tprs, scores = [], [], []

    

for (train, test), i in zip(cv.split(X, y), range(5)):

    clf.fit(X.iloc[train], y.iloc[train])

    _, _, auc_score_train = compute_roc_auc(train, clf)

    fpr, tpr, auc_score = compute_roc_auc(test, clf)

    scores.append((auc_score_train, auc_score))

    fprs.append(fpr)

    tprs.append(tpr)



plot_roc_curve(fprs, tprs);

pd.DataFrame(scores, columns=['AUC Train', 'AUC Test'])


clf.predict(X.iloc[0:10])