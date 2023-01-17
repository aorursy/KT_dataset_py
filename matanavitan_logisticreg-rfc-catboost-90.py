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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/online-shoppers-intention/online_shoppers_intention.csv')
df
df.isna().sum()
df.info()
# We have enough data, we don't have to impute the missing values,
# we can just drop them.
df.dropna(inplace=True)
df.isna().sum()
df.nunique()
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(df, test_size=0.2)
y_train, y_test = X_train.pop('Revenue'), X_test.pop('Revenue')
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression

categorical_features = ['SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType',
                        'Weekend']
ohe = OneHotEncoder(handle_unknown='ignore')
transformer = make_column_transformer((ohe, categorical_features))
clf = LogisticRegression()
basic_pipe = make_pipeline(transformer, clf)
basic_pipe.fit(X_train, y_train)
basic_pipe.score(X_test, y_test)
# Let's checkout correlation between features and the label
df.corr()['Revenue']
# It seems llike there is high corr between PageValues and the Revenue
# From my experience RFC works very well for classification puposes, let's check it out
# I also use GridSearch for hyperparameters optimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


categorical_features = ['SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
                        'VisitorType', 'Weekend']
numerical_features = ['Administrative','Administrative_Duration','Informational','Informational_Duration',
                      'ProductRelated','ProductRelated_Duration','BounceRates','ExitRates',
                      'PageValues','SpecialDay']
ohe = OneHotEncoder(handle_unknown='ignore')
ss = StandardScaler()
transformer = make_column_transformer((ohe, categorical_features),
                                      (ss, numerical_features),
                                      remainder='passthrough')
clf = RandomForestClassifier()
rfc_pipe = Pipeline([('transformer', transformer), 
                    ('rf', clf)])

# Lets optimize some hyperparams
params = {'rf__n_estimators':[100,150,200],
          'rf__max_depth':[None, 30,50]}
grid_pipe = GridSearchCV(rfc_pipe, param_grid=params, cv=5)
grid_pipe.fit(X_train, y_train)
print(grid_pipe.best_params_)
print(grid_pipe.best_score_)
encoded_X_train = grid_pipe.best_estimator_.get_params()['transformer'].transform(X_train)
encoded_X_train
# Let's visualize the feature importance of the forest

best_forest = grid_pipe.best_estimator_.get_params()['rf']
importances = best_forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in best_forest],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(encoded_X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(encoded_X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
# Let's plot only 17 features
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()
encoded_X_train = grid_pipe.best_estimator_.get_params()['transformer'].transform(X_train)
encoded_X_train
encoded_X_test = grid_pipe.best_estimator_.get_params()['transformer'].transform(X_test)
# Let's try to use only those best features for training a new RFC
best_features_list = []
score_list = [0]
# Let's find the ideal amount of feature between the first 30
for i in range(1, 30):
    amount_of_selected_features = i
    for f in range(amount_of_selected_features):
        best_features_list.append(indices[f])

    rfc = RandomForestClassifier(max_depth=30, n_estimators=100)
    rfc.fit(encoded_X_train[:, best_features_list], y_train)

    score_list.append(rfc.score(encoded_X_test[:, best_features_list],
                                y_test))
# A comment, this is very unstable search because of the nature of RFC.
# You can't really relay on one iteration but for simplicity let's do it for one.
print(f'best amount of features: {np.argmax(score_list)}.'
      f'score: {np.max(score_list).round(5)}')
# Using all of the features
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
endoded_train_label = le.fit_transform(y_train)
endoded_test_label = le.fit_transform(y_test)

transformer = make_column_transformer((ohe, categorical_features),
                                      (ss, numerical_features),
                                      remainder='passthrough')

model = CatBoostClassifier(iterations=20,
                           depth=2,
                           learning_rate=1,
                           loss_function='CrossEntropy',
                           verbose=False)

catboost_pipe = Pipeline([('transformer', transformer), 
                         ('cb', model)])


# Lets optimize some hyperparams
params = {'cb__iterations':[10,20],
          'cb__depth':[2, 3, 4],
         'cb__learning_rate': np.linspace(0.75, 1, 3)}

cb_pipe = GridSearchCV(catboost_pipe, param_grid=params, cv=5)

#train the model
cb_pipe.fit(X_train, endoded_train_label)

print(cb_pipe.best_params_)
print(cb_pipe.best_score_)
# Select Best features according to f_classif test, 
# with the best amount of features from the previous step.
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

le = LabelEncoder()
endoded_train_label = le.fit_transform(y_train)
endoded_test_label = le.transform(y_test)

transformer = make_column_transformer((ohe, categorical_features),
                                      (ss, numerical_features),
                                      remainder='passthrough')

model = CatBoostClassifier(iterations=10,
                           depth=3,
                           learning_rate=0.75,
                           loss_function='CrossEntropy',
                           verbose=False)
bestk = SelectKBest(score_func=f_classif, k=np.argmax(score_list))

catboost_pipe = Pipeline([('transformer', transformer), 
                          ('bestk', bestk),
                         ('cb', model)])

#train the model
catboost_pipe.fit(X_train, endoded_train_label)

print(catboost_pipe.score(X_test, endoded_test_label))
# Conclusions:
# It's looks like taking the best amount of features from RFC,
# and then select by f_classif this amount of features gave us the best score
