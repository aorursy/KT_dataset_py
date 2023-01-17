import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/heart-disease/heart.csv')

data.head()
data.describe()
data.isnull().values.any()
import seaborn as sns
_ = data.corr()

sns.heatmap(_, cmap=sns.diverging_palette(150, 275, s=80, l=55, n=9))

#Not the prettiest plot but I'm using a diverging colour pallette so 

#that its easier to pick out the values closest to 0 around the target
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

X = data.drop(['target'], axis=1)

y = data['target']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1, test_size=.2)
from sklearn.ensemble import GradientBoostingRegressor



est_range = range(5, 105, 5)

score_graph = {'ROC AUC Score': [], 'n_estimators': []}

est_list = []

score_list = []

for num in est_range:

    gbr = GradientBoostingRegressor(n_estimators=num, random_state=0)

    gbr.fit(train_X, train_y)

    ls_preds = gbr.predict(val_X)

    acc = roc_auc_score(val_y, ls_preds)

    print('ROC AUC Score with',num ,'estimators is: ', acc)

    score_graph['ROC AUC Score'] = score_graph['ROC AUC Score'] + [acc]

    score_graph['n_estimators'] = score_graph['n_estimators'] + [num]
sns.lineplot(x=score_graph['n_estimators'], y=score_graph['ROC AUC Score'])



# When selecting an optimal parameter where there is a plateau

# I tend to pick the side of the plateau that has the lowest complexity.

# This helps prevent overfitting
import warnings

warnings.filterwarnings("ignore")
import eli5

from eli5.sklearn import PermutationImportance



my_model = GradientBoostingRegressor(n_estimators=35).fit(train_X, train_y)



perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
feat = ['ca', 'oldpeak', 'thal']

X = data[feat]

y = data['target']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1, test_size=.2)
gbr = GradientBoostingRegressor(n_estimators=35, random_state=0)

gbr.fit(train_X, train_y)

ls_preds = gbr.predict(val_X)

print('ROC AUC Scoree is: ', roc_auc_score(val_y, ls_preds))
est_range = range(5, 105, 5)

score_graph = {'ROC AUC Score': [], 'n_estimators': []}

est_list = []

score_list = []

# A coder with some forsight might have just created a function for this instead of copy pasting my prior code

for num in est_range:

    gbr = GradientBoostingRegressor(n_estimators=num, random_state=0)

    gbr.fit(train_X, train_y)

    ls_preds = gbr.predict(val_X)

    acc = roc_auc_score(val_y, ls_preds)

    print('ROC AUC Score with',num ,'estimators is: ', acc)

    score_graph['ROC AUC Score'] = score_graph['ROC AUC Score'] + [acc]

    score_graph['n_estimators'] = score_graph['n_estimators'] + [num]

sns.lineplot(x=score_graph['n_estimators'], y=score_graph['ROC AUC Score'])
gbr = GradientBoostingRegressor(n_estimators=5, random_state=0)

gbr.fit(train_X, train_y)

ls_preds = gbr.predict(val_X)

print('ROC AUC Scoree is: ', roc_auc_score(val_y, ls_preds))
from sklearn.metrics import confusion_matrix



def rounder(num, thresh=0.5):

    if num >= thresh:

        return 1

    else:

        return 0
rounding = pd.Series([rounder(x,.2) for x in ls_preds])

confuse = confusion_matrix(val_y, rounding)

print(confuse[[1],[0]])
import matplotlib

from matplotlib.pyplot import figure

false_negitives = []

threshholds = []

for num in range(1, 10, 1):

    n = num/10

    rounding = pd.Series([rounder(x,n) for x in ls_preds])

    print('Accuracy with threshold set to',n ,'is: ', accuracy_score(val_y, rounding))

    confuse = confusion_matrix(val_y, rounding)

    false_negitives = false_negitives + [int(confuse[[1],[0]])]

    threshholds = threshholds + [n]

    confuse = pd.DataFrame(confuse)

    figure(num=None, figsize=(5, 5))

    sns.heatmap(confuse, linewidths=1,annot=True, fmt='.5g', annot_kws={"size": 12},cmap="YlGnBu", 

                yticklabels=['Negitive', 'Positive'], xticklabels=['Negitive Predicted', 'Positive Predicted']).set_title('Threshhold set to: '+str(n))
# this plots the rate of false negitives as we increase the 

# threshholds for when a person is determined to have heart deasease 

sns.lineplot(y=false_negitives, x=threshholds)