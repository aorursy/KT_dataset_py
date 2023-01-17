# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

from matplotlib import pyplot as plt

import numpy as np
from sklearn.metrics import f1_score,make_scorer
from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score,RepeatedStratifiedKFold

from catboost import CatBoostClassifier

from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from scipy.stats import ttest_1samp,ttest_rel
import statsmodels.stats.api as sms
data = pd.read_csv("../input/mf-accelerator/contest_train.csv")
target = data.TARGET
data = data.fillna(0)
features = data.drop(columns=["TARGET","ID"])
features.head()
scorer = make_scorer(f1_score,average='macro')
rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=5,
      random_state=12)
%%time
clf_cb = CatBoostClassifier(task_type='GPU',random_state=100, loss_function='MultiClass',
                            auto_class_weights="Balanced",iterations=2500,verbose=2500)  
 
scores = cross_val_score(clf_cb,features,target,scoring=scorer,cv=rskf)
scores
print("Statistics score")
print("-------")
print(f"Mean: {scores.mean()}")
print(f"Std: {scores.std(ddof=1)}")
print(f"Max: {scores.max()}")
print(f"Min: {scores.min()}")
print("-------")
%%time
clf_cb2 = CatBoostClassifier(task_type='GPU',random_state=100, loss_function='MultiClassOneVsAll',
                            auto_class_weights="Balanced",iterations=1050,
                           depth=7, l2_leaf_reg=1,verbose=1050)  
 
scores2 = cross_val_score(clf_cb2,features,target,scoring=scorer,cv=rskf)
scores2
print("Statistics score2")
print("-------")
print(f"Mean: {scores2.mean()}")
print(f"Std: {scores2.std(ddof=1)}")
print(f"Max: {scores2.max()}")
print(f"Min: {scores2.min()}")
print("-------")
qqplot(scores,line = 's');
(s1,p1) = shapiro(scores)
print(f"The Shapiro-Wilk statistic value: {s1}")
print(f"p-value: {p1}")
qqplot(scores2,line = 's');
(s2,p2) = shapiro(scores2)
print(f"The Shapiro-Wilk statistic value: {s2}")
print(f"p-value: {p2}")
ttest_rel(scores, scores2, axis=0)
cm = sms.CompareMeans(sms.DescrStatsW(scores), sms.DescrStatsW(scores2))
print(cm.tconfint_diff(usevar='unequal'))
