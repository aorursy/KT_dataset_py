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
dataset = pd.read_csv('/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv')

cols = dataset.columns

factors = dataset.iloc[:, 0:13]

target = dataset.iloc[:,13:] 
import seaborn as sns
corr = factors.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
#transformation

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_factors = scaler.fit_transform(factors)
#SGD Classifier

from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss="log", max_iter=10).fit(scaled_factors, target)
from sklearn.feature_selection import RFE

rfe_s = RFE(clf, n_features_to_select=5, step=1)

rfe_s = rfe_s.fit(scaled_factors, target)

rank = rfe_s.ranking_

support = rfe_s.support_

print(rank)

print(support)
from sklearn.feature_selection import SelectFromModel

sfm_s = SelectFromModel(clf, prefit=True)

support2 = sfm_s.get_support()

print(support2)
#Evaluate the first model with all vars applied

from sklearn.model_selection import cross_val_score

scores1 = cross_val_score(clf, scaled_factors, target, cv=5)

scores1 = np.mean(scores1)

print("First model:", scores1)
#Let's refractor the factors column with only 8 variables

factors_updated = pd.DataFrame({'age': scaled_factors[:, 0], 'cp': scaled_factors[:, 2],

                               'trestbps': scaled_factors[:, 3],'thalach': scaled_factors[:, 7],

                               'exang': scaled_factors[:, 8], 'oldpeak': scaled_factors[:, 9],

                               'slope': scaled_factors[:, 10], 'thal': scaled_factors[:, 12]})

print(factors_updated)
clf2 = SGDClassifier(loss="log", max_iter=10).fit(factors_updated, target)

scores2 = cross_val_score(clf2, factors_updated, target, cv=5)

scores2 = np.mean(scores2)

print("Second model:", scores2)

