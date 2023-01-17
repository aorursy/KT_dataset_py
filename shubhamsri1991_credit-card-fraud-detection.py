# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
credit_card_dataset = pd.read_csv('../input/creditcardfraud/creditcard.csv')
credit_card_dataset.head(10).T
credit_card_dataset.shape
credit_card_dataset.info()
credit_card_dataset.describe()
credit_card_dataset.isnull().sum()
credit_card_dataset.duplicated().sum()
unique = credit_card_dataset.nunique()
unique
credit_card_dataset.drop_duplicates(keep='first', inplace=True)
credit_card_dataset.shape
variance = credit_card_dataset.std()/credit_card_dataset.mean()
variance
plt.figure(figsize=(30,20))
corr = credit_card_dataset.corr()
sns.heatmap(corr, annot=True, cmap='RdYlGn')
plt.show()
features = credit_card_dataset.drop(['Class'], axis=1)
label = credit_card_dataset['Class']
from sklearn.feature_selection import f_classif
fval, pval = f_classif(features, label)
for i in range(len(pval)):
    print(features.columns[i], pval[i].round(4))
from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreeRegressor()
etr.fit(features, label)
plt.figure(figsize=(12,5))
features_importance = pd.Series(etr.feature_importances_, index=features.columns)
features_importance.nlargest(20).plot(kind = 'barh')
plt.show()
count_classes = pd.value_counts(credit_card_dataset['Class'], sort=True)
plt.figure(figsize=(12,5))
count_classes.plot(kind='bar', rot=0)
plt.title('Outcome Class Distribution')
plt.xlabel("Class")
plt.ylabel("No Of Count")
plt.show()
normal_trans = credit_card_dataset[credit_card_dataset['Class']==0]
fraud_trans = credit_card_dataset[credit_card_dataset['Class']==1]
normal_trans.shape, fraud_trans.shape
features = credit_card_dataset.drop(['Class'], axis=1)
label = credit_card_dataset['Class']
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=20)
features_new, label_new = smt.fit_sample(features, label)
features_new.shape, label_new.shape
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=20, shuffle=False, random_state=None)
skf.get_n_splits(features_new, label_new)
for train_index, test_index in skf.split(features_new, label_new):
    features_train, features_test = features_new.iloc[train_index], features_new.iloc[test_index]
    label_train, label_test = label_new.iloc[train_index], label_new.iloc[test_index]
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(n_estimators=20, max_depth=8, min_samples_leaf=10,
                                            min_samples_split=30, random_state=5, oob_score=True)
random_forest_model.fit(features_train, label_train)
random_forest_model.predict(features_test)
from sklearn.metrics import recall_score
recall_score(label_train, random_forest_model.predict(features_train))
from sklearn.metrics import recall_score
recall_score(label_test, random_forest_model.predict(features_test))
from sklearn.metrics import precision_score
precision_score(label_train, random_forest_model.predict(features_train))
from sklearn.metrics import precision_score
precision_score(label_test, random_forest_model.predict(features_test))
from sklearn.metrics import accuracy_score
accuracy_score(label_train, random_forest_model.predict(features_train))
from sklearn.metrics import accuracy_score
accuracy_score(label_test, random_forest_model.predict(features_test))