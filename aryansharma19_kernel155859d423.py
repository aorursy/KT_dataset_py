# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/Glass_Quality_Participants_Data/Train.csv')
test = pd.read_csv('/kaggle/input/Glass_Quality_Participants_Data/Test.csv')
SUBMISSION = pd.read_excel('/kaggle/input/Glass_Quality_Participants_Data/Sample_Submission.xlsx')
data.head()
sns.heatmap(data.corr(), cmap = 'RdYlGn')
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin

class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=10.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh
        
        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        #if impute:
        #    self.imputer = Imputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        #if hasattr(self, 'imputer'):
        #    X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=10.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped=True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped=True
        return X
tf = ReduceVIF()
x = tf.fit_transform(data.drop(['class'], axis = 1), data['class'])
test.drop(['pixel_area', 'ymax', 'xmin', 'grade_A_Component_2'], axis = 1, inplace = True)
sns.heatmap(pd.concat([x, data['class']], axis = 1).corr(), cmap = 'RdYlGn')
x.head()
x['new_x'] = x[['x_component_1', 'x_component_2', 'x_component_4', 'x_component_5']].sum(axis = 1)/4
x.head()
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns = x.columns)
x.head()
cols = x.columns
sns.heatmap(x.corr(), cmap = 'RdYlGn')
y = data['class']
x.shape, y.shape
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression

#skf = StratifiedShuffleSplit(random_state = 101)
skf = StratifiedKFold(n_splits = 5, random_state = 101, shuffle = True)

lr = LogisticRegression()
mnb = MultinomialNB()
bernoulli = BernoulliNB()
rfc = RandomForestClassifier()
ada = AdaBoostClassifier()
bag = BaggingClassifier()
etc = ExtraTreesClassifier()
gbc = GradientBoostingClassifier()
knn = KNeighborsClassifier()
xgb = XGBClassifier()
cat = CatBoostClassifier(silent = True)
lgb = LGBMClassifier()

models = [(lr, 'Logistic Regression'), (bernoulli, 'BernoulliNB'),(rfc, 'rfc'),(ada, 'ada'),
          (bag, 'Baggingclf'), (etc, 'etc'), (gbc, 'GBC'),
          (knn, 'knn'), (xgb, 'xgb'), (cat, 'cat'), (lgb, 'lgb')]

loss = {
    'Logistic Regression' : [],
    'BernoulliNB' : [],
    'rfc' : [],
    'ada' : [],
    'Baggingclf' : [],
    'etc' : [],
    'GBC' : [],
    'knn' : [],
    'xgb' : [],
    'lgb' : [],
    'cat' : []
}
smote = SMOTE()
x, y = smote.fit_sample(x, y)

for model, label in models:
    l = []
    for train_idx, test_idx in skf.split(x, y):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(x_train, y_train)
        pred = model.predict_proba(x_test)[::1]
        l.append(log_loss(y_test, pred))
    loss[label].extend(l)
    print(f'{label} is ready !!')
result = pd.DataFrame(loss)
result
result.mean(axis = 0)
plt.figure(figsize = (15, 5))
plt.subplot(1, 2, 1)
pd.Series(cat.feature_importances_, index = x_train.columns).plot(kind = 'barh', title = 'CATBOOST')
plt.subplot(1, 2, 2)
pd.Series(etc.feature_importances_, index = x_train.columns).plot(kind = 'barh', title = 'EXTRA TREE CLF')
plt.tight_layout()
test.head()
test['new_x'] = test[['x_component_1', 'x_component_2', 'x_component_4', 'x_component_5']].sum(axis = 1)/4
test = pd.DataFrame(scaler.transform(test), columns = cols)
submission = etc.predict_proba(test)[::1]
submission = pd.DataFrame(submission, columns = ['1', '2'])
submission.head()

SUBMISSION.iloc[:, :] = submission.values
SUBMISSION

SUBMISSION.to_excel('VIF_robust_scaler_smote_newX_excess_removal_comp_ETC.xlsx', index = False)
