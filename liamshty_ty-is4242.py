# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('../input/train.csv')
df.head()
df1 = pd.get_dummies(df, prefix=['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat',
       'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat',
       'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
       'ps_car_10_cat', 'ps_car_11_cat'],columns=['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat',
       'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat',
       'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
       'ps_car_10_cat', 'ps_car_11_cat'])
df1

df2 = df1.filter(regex='_-1')
df2.head()
list(df2)
df1.loc[df.ps_ind_02_cat==-1, df1.columns.str.startswith("ps_ind_02_cat")] = np.nan
df1.loc[df.ps_ind_04_cat==-1, df1.columns.str.startswith("ps_ind_04_cat")] = np.nan
df1.loc[df.ps_ind_05_cat==-1, df1.columns.str.startswith("ps_ind_05_cat")] = np.nan
df1.loc[df.ps_car_01_cat==-1, df1.columns.str.startswith("ps_car_01_cat")] = np.nan
df1.loc[df.ps_car_02_cat==-1, df1.columns.str.startswith("ps_car_02_cat")] = np.nan
df1.loc[df.ps_car_03_cat==-1, df1.columns.str.startswith("ps_car_03_cat")] = np.nan
df1.loc[df.ps_car_04_cat==-1, df1.columns.str.startswith("ps_car_04_cat")] = np.nan
df1.loc[df.ps_car_05_cat==-1, df1.columns.str.startswith("ps_car_05_cat")] = np.nan
df1.loc[df.ps_car_06_cat==-1, df1.columns.str.startswith("ps_car_06_cat")] = np.nan
df1.loc[df.ps_car_07_cat==-1, df1.columns.str.startswith("ps_car_07_cat")] = np.nan
df1.loc[df.ps_car_09_cat==-1, df1.columns.str.startswith("ps_car_09_cat")] = np.nan
df1.loc[df.ps_car_10_cat==-1, df1.columns.str.startswith("ps_car_10_cat")] = np.nan
df1.loc[df.ps_car_11_cat==-1, df1.columns.str.startswith("ps_car_11_cat")] = np.nan
df1 = df1.drop(columns=['ps_ind_02_cat_-1',
 'ps_ind_04_cat_-1',
 'ps_ind_05_cat_-1',
 'ps_car_01_cat_-1',
 'ps_car_02_cat_-1',
 'ps_car_03_cat_-1',
 'ps_car_05_cat_-1',
 'ps_car_07_cat_-1',
 'ps_car_09_cat_-1'])
df1.info()
labels = df1.columns[2:]
X = df1[labels]
y = df1['target']

target_count = df1.target.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (target)')
# 2
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold 
imp_1 = SimpleImputer(missing_values=np.nan, strategy='mean')

# pipeline 1
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
import warnings 
warnings.filterwarnings('ignore')
imp_1 = SimpleImputer(missing_values=np.nan, strategy='mean')
KNN_1 = KNeighborsClassifier()
pipe_1 =  Pipeline([('inpute', imp_1), ('KNN1', KNN_1)])
grid_search = GridSearchCV(pipe_1, {'KNN1__n_neighbors': [1,3,5]},cv=5)
grid_search.fit(X.sample(frac=0.05), y.sample(frac=0.05))
grid_search.best_params_

from scipy import interp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
cv = StratifiedKFold(n_splits=5)
classifier = pipe_1

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X.sample(frac=0.05), y.sample(frac=0.05)):
    probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Pipeline 1')
plt.legend(loc="lower right")
plt.show()
#pipeline 2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
import warnings 
warnings.filterwarnings('ignore')
imp_1 = SimpleImputer(missing_values=np.nan, strategy='mean')
kbest = SelectKBest(f_classif)
pipe_2 = Pipeline([('inpute', imp_1),('kbest', kbest), ('lr', LogisticRegression(solver='lbfgs'))])
grid_search = GridSearchCV(pipe_2, {'kbest__k': [1,2,3,4], 'lr__C': np.logspace(-10, 10, 5)},cv=5)
grid_search.fit(X.sample(frac=0.05), y.sample(frac=0.05))
grid_search.best_params_
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
cv = StratifiedKFold(n_splits=5)
classifier = pipe_2

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X.sample(frac=0.05), y.sample(frac=0.05)):
    probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Pipeline 2')
plt.legend(loc="lower right")
plt.show()
#pipeline 3
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
pca = PCA()
pipe_3 = Pipeline([('inpute', imp_1),('pca', pca),('vts',VarianceThreshold(threshold=(.8 * (1 - .8)))),('lr', LogisticRegression(solver='lbfgs'))])
pipe_3.get_params().keys()
grid_search = GridSearchCV(pipe_3, {'pca__n_components':[2,4,6],'lr__C': np.logspace(-10, 10, 5)},cv=5)
grid_search.fit(X.sample(frac=0.05), y.sample(frac=0.05))
grid_search.best_params_
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
cv = StratifiedKFold(n_splits=5)
classifier = pipe_3

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X.sample(frac=0.05), y.sample(frac=0.05)):
    probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Pipeline 3')
plt.legend(loc="lower right")
plt.show()
#pipeline 4
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
pca = PCA()
pipe_4 = Pipeline([('inpute', imp_1),('pca', pca),('dst', DecisionTreeClassifier())])
pipe_4.get_params().keys()
grid_search = GridSearchCV(pipe_4, {'pca__n_components':[2,4,6]},cv=5)
grid_search.fit(X.sample(frac=0.01), y.sample(frac=0.01))
grid_search.best_params_
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
cv = StratifiedKFold(n_splits=5)
classifier = pipe_4

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X.sample(frac=0.05), y.sample(frac=0.05)):
    probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Pipeline 4')
plt.legend(loc="lower right")
plt.show()
# 3
from imblearn.over_sampling import SMOTE
pipe_3_smote = Pipeline([('inpute', imp_1),('pca', pca),('vts',VarianceThreshold(threshold=(.8 * (1 - .8)))),('lr', LogisticRegression(solver='lbfgs'))])
pipe_3_smote.steps.insert(1,['smote',SMOTE()])

print(pipe_3_smote)
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
cv = StratifiedKFold(n_splits=5)
classifier = pipe_3_smote

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X.sample(frac=0.05), y.sample(frac=0.05)):
    probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Pipeline 3 SMOTE')
plt.legend(loc="lower right")
plt.show()
from imblearn.under_sampling import ClusterCentroids
pipe_3_under = Pipeline([('inpute', imp_1),('pca', pca),('vts',VarianceThreshold(threshold=(.8 * (1 - .8)))),('lr', LogisticRegression(solver='lbfgs'))])
pipe_3_under.steps.insert(1,['under',ClusterCentroids()])
pipe_3_under
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
cv = StratifiedKFold(n_splits=5)
classifier = pipe_3_under

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X.sample(frac=0.05), y.sample(frac=0.05)):
    probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Pipeline 3 Under')
plt.legend(loc="lower right")
plt.show()