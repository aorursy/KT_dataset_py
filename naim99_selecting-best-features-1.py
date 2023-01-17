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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

df = pd.read_excel('../input/lr-labels/LR_label.xlsx')

X = df.iloc[: , :-1]

Y = df.iloc[: , -1]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(df['Q14'])

X = pd.get_dummies(df.drop('Q14', axis=1))



from sklearn.model_selection import  cross_val_score, StratifiedKFold

from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier(max_depth = 3, min_samples_split = 2)



dt.fit(X, y)

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,

            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,

            min_samples_split=2, min_weight_fraction_leaf=0.0,

            presort=False, random_state=None, splitter='best')



pd.Series(y).value_counts()



C0 = np.linspace(0,1)

C1 = 1.0 - C0



gini = 1 - C0**2 - C1**2



plt.plot(C0, gini)

plt.title('Gini index for a Binary Classification')

plt.xlabel('Fraction of samples in class 0')

plt.ylabel('Gini index')

plt.show() 



feature_importances = pd.DataFrame(dt.feature_importances_,

                                   index = X.columns,

                                    columns=['importance']).sort_values('importance',

                                                                        ascending=False)

feature_importances.head()





from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier



rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1)

et = ExtraTreesClassifier(class_weight='balanced', n_jobs=-1)



rf.fit(X, y)



RandomForestClassifier(bootstrap=True, class_weight='balanced',

            criterion='gini', max_depth=None, max_features='auto',

            max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)



all(rf.feature_importances_ == np.mean([tree.feature_importances_ for tree in rf.estimators_], axis=0))



importances = rf.feature_importances_

std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)



indices = np.argsort(importances)[::-1]

feature_names = X.columns

# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)

plt.xlim([-1, X.shape[1]])



et.fit(X, y)



importances = et.feature_importances_

std = np.std([tree.feature_importances_ for tree in et.estimators_], axis=0)



indices = np.argsort(importances)[::-1]

feature_names = X.columns



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)

plt.xlim([-1, X.shape[1]])



dt = DecisionTreeClassifier()

dt.fit(X, y)



importances = pd.DataFrame(zip(dt.feature_importances_,

                               rf.feature_importances_,

                               et.feature_importances_),

                           index=X.columns,

                           columns=['dt_importance',

                                    'rf_importance',

                                    'et_importance']).sort_values('rf_importance',

                                                                   ascending=False)





importances.plot(kind='bar')

importances.head()


