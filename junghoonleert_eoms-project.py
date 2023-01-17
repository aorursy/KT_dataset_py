import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Loading data, dividing, modeling and EDA below



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



# Data Processing for cluster1_0



data = pd.read_csv('../input/eoms-project.csv')

y10 = data.cluster1_0

feature_names = [i for i in data.columns if data[i].dtype in [np.float64]]

X = data[feature_names] 

train_X, val_X, train_y10, val_y10 = train_test_split(X, y10, random_state=1)

my_model = RandomForestClassifier(n_estimators=100,

                                  random_state=0).fit(train_X, train_y10)
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y10)

eli5.show_weights(perm, top = 50, feature_names = val_X.columns.tolist(), show_feature_values = True)
# Data Processing for cluster1_1



y11 = data.cluster1_1

train_X, val_X, train_y11, val_y11 = train_test_split(X, y11, random_state=1)

my_model = RandomForestClassifier(n_estimators=100,

                                  random_state=0).fit(train_X, train_y11)

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y11)

eli5.show_weights(perm, top = 50, feature_names = val_X.columns.tolist(), show_feature_values = True)
# Data Processing for cluster1_2



y12 = data.cluster1_2

train_X, val_X, train_y12, val_y12 = train_test_split(X, y12, random_state=1)

my_model = RandomForestClassifier(n_estimators=100,

                                  random_state=0).fit(train_X, train_y12)

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y12)

eli5.show_weights(perm, top = 50, feature_names = val_X.columns.tolist(), show_feature_values = True)
# Data Processing for cluster2_1



y21 = data.cluster2_1

train_X, val_X, train_y21, val_y21 = train_test_split(X, y21, random_state=1)

my_model = RandomForestClassifier(n_estimators=100,

                                  random_state=0).fit(train_X, train_y21)

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y21)

eli5.show_weights(perm, top = 50, feature_names = val_X.columns.tolist(), show_feature_values = True)
# Data Processing for cluster2_2



y22 = data.cluster2_2

train_X, val_X, train_y22, val_y22 = train_test_split(X, y22, random_state=1)

my_model = RandomForestClassifier(n_estimators=100,

                                  random_state=0).fit(train_X, train_y22)

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y22)

eli5.show_weights(perm, top = 50, feature_names = val_X.columns.tolist(), show_feature_values = True)