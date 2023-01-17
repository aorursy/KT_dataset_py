
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

from collections import Counter

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from imblearn.datasets import make_imbalance
from imblearn.under_sampling import RandomUnderSampler

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import mlcrate as mlc
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

pal = sns.color_palette()

print('# File sizes')
for f in os.listdir('../input'):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
dtypes = {'ip': 'int32', 'app':'int16', 'device': 'int16', 'os': 'int16', 'channel': 'int16'}
#import first 10,000,000 rows of train and all test data
train = pd.read_csv('../input/train_sample.csv', parse_dates=['click_time', 'attributed_time'], 
                    dtype=dtypes)
test = pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['click_time'])

train.head()
train.describe()
from sklearn.preprocessing import scale
y_train = train['is_attributed']
x_train = scale(train.drop(['is_attributed', 'click_time', 'attributed_time'], axis=1))
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.metrics import classification_report_imbalanced
from collections import Counter
print("Training class distribution summary: {}".format(Counter(y_train)))
from scipy import interp 
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
                              BaggingClassifier, VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline
LW = 2
RANDOM_STATE = 42
cv = StratifiedKFold(n_splits=2)
# Kneighbor parameterers
kn_params = {'n_neighbors': 5, 'n_jobs': -1}

mlp_params = {'alpha': 1}

rf_params = {
    'n_jobs': -1,
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

et_params = {
    'n_jobs': -1,
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'verbose': 0
}

ada_params = {'n_estimators': 100, 'learning_rate': 0.75}

gb_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}
classifiers = [
    ('5NN', KNeighborsClassifier(**kn_params)), 
    ('Bagging', BaggingClassifier()),
    ('MLP', MLPClassifier(**mlp_params)),
    ('forest', RandomForestClassifier(**rf_params)),
    ('extra_trees', ExtraTreesClassifier(**et_params)),
    ('adaboost', AdaBoostClassifier(**ada_params)),
    ('gboost', GradientBoostingClassifier(**gb_params))
]
samplers = [['ADASYN', ADASYN(random_state=RANDOM_STATE, n_jobs=-1, n_neighbors=5)]]
pipelines = [[
    '{}-{}'.format(sampler[0], classifier[0]),
    make_pipeline(sampler[1], classifier[1])
] for sampler in samplers for classifier in classifiers]
pipelines
from time import time
%%time 
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(1, 1, 1)

for name, pipeline in pipelines:
    start = time()
    mean_tpr  = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for train, test in cv.split(x_train, y_train):
        probas_ = pipeline.fit(x_train[train], y_train[train]).predict_proba(x_train[test])
        fpr, tpr, thresholds = roc_curve(y_train[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        
        
    mean_tpr /= cv.get_n_splits(x_train, y_train)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, linestyle='--', label='{} (area = %0.2f)'.format(name) % mean_auc, lw=LW)
    total_time = time() - start
    print('{} took {} seconds'.format(name, total_time))
    
    
plt.plot([0, 1], [0, 1], linestyle='--', lw=LW, color='k', label='Luck')

# Make nice plotting
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

