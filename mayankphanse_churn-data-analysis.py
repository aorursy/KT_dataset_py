!pip install -U imbalanced-learn
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



        

import matplotlib.pyplot as plt

from matplotlib import cm

import seaborn as sns; sns.set()

%matplotlib inline





from sklearn.tree import DecisionTreeClassifier





from imblearn.pipeline import make_pipeline, Pipeline

from imblearn.over_sampling import SMOTE



from mlxtend.plotting import plot_decision_regions

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import KBinsDiscretizer, FunctionTransformer

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler

from category_encoders import WOEEncoder

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier





from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC

from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, FunctionTransformer

from category_encoders import OneHotEncoder



from mlxtend.evaluate import feature_importance_permutation

from sklearn.model_selection import train_test_split



from mlxtend.feature_extraction import PrincipalComponentAnalysis

from mlxtend.preprocessing import standardize



from mlxtend.plotting import plot_pca_correlation_graph





from mlxtend.plotting import plot_decision_regions

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import KBinsDiscretizer, FunctionTransformer

from sklearn.model_selection import cross_val_score



from sklearn.pipeline import make_pipeline, Pipeline





import warnings

warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv')

df_orig = pd.read_csv('../input/churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv')

target = 'churn'

y = df[target]

labels = df.columns
df.head()
df.info()
df = df.drop(['phone number'], axis = 1)

df = df.drop(['area code'], axis = 1)

df = df.drop(['state'], axis = 1)
df.head()
df['international plan'] = df['international plan'].map({'yes': 1, 'no': 0})

df['voice mail plan'] = df['voice mail plan'].map({'yes': 1, 'no': 0})

df['churn'] = df['churn'].map({True: 1, False: 0})
df.head()
X = df.drop([target],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
scaler = MinMaxScaler()

lr = LogisticRegression()

pipe = make_pipeline(scaler, lr)



pipe.fit(X_train, y_train)



train_preds = pipe.predict(X_train)

test_preds = pipe.predict(X_test)
scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')

print(scores.mean(), "+/-", scores.std())
def stringify(data):

    df = pd.DataFrame(data)

    for c in df.columns.tolist():

        df[c] = df[c].astype(str)

    return df



binner = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

objectify = FunctionTransformer(func=stringify, 

                                validate=False)

clf = LogisticRegression(class_weight='balanced')

encoder = WOEEncoder()

scorecard = make_pipeline(binner, objectify, encoder, lr)





scores = cross_val_score(scorecard, X, y, cv=5, scoring='roc_auc')

print(scores.mean(), "+/-", scores.std())
X = Pipeline(scorecard.steps[:-1]).fit_transform(X, y).values

used_cols = [c for c in df.columns.tolist() if c not in [target]]



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=1, stratify=y)



clf.fit(X_train, y_train)

imp_vals, imp_all = feature_importance_permutation(

    predict_method=clf.predict, 

    X=X_test,

    y=y_test,

    metric='accuracy',

    num_rounds=10,

    seed=1)



std = np.std(imp_all, axis=1)

indices = np.argsort(imp_vals)[::-1]



plt.figure()

plt.title("Scorecard Feature Importance via Permutation Importance")

plt.bar(range(X.shape[1]), imp_vals[indices],

        yerr=std[indices])

# plt.xticks(range(X.shape[1]), indices)

plt.xticks(range(X.shape[1]), np.array(used_cols)[indices], rotation = 90)

plt.xlim([-1, X.shape[1]])

plt.ylim([0, 0.05])

plt.show()
important_feat = ['customer service calls', 'total day minutes','total intl calls']
from sklearn.base import TransformerMixin



class ForestEncoder(TransformerMixin):

    

    def __init__(self, forest):

        self.forest = forest

        self.n_trees = 1

        try:

            self.n_trees = self.forest.n_estimators

        except:

            pass

        self.ohe = OneHotEncoder(cols=range(self.n_trees), use_cat_names=True)

        

    def fit(self, X, y=None):

        self.forest.fit(X, y)

        self.ohe.fit(self.forest.apply(X))

        return self

    

    def transform(self, X, y=None):

        return self.ohe.transform(self.forest.apply(X))
#entropy criterion

used_cols = [c for c in df.columns.tolist() if c not in [target]]

X, y = df[used_cols].values, df[target].values



N = 5



rf = RandomForestClassifier(max_depth = N, n_estimators=100, n_jobs=-1, random_state=42,criterion = 'entropy', max_leaf_nodes = 2**N-1)

encoder = ForestEncoder(rf)

clf = LogisticRegression(class_weight='balanced')

pipe = make_pipeline(encoder, clf)

pipe.fit(X, y)



scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')

print(scores.mean(), "+/-", scores.std())
#gini criterion



rf = RandomForestClassifier(max_depth = N, n_estimators=100, n_jobs=-1, random_state=42,criterion = 'gini', max_leaf_nodes = 2**N-1)

encoder = ForestEncoder(rf)

clf = LogisticRegression(class_weight='balanced')

pipe = make_pipeline(encoder, clf)

pipe.fit(X, y)



scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')

print(scores.mean(), "+/-", scores.std())
encoder = ForestEncoder(rf)

clf = LogisticRegression(class_weight='balanced')

pipe = make_pipeline(encoder, clf)



scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')

print(scores.mean(), "+/-", scores.std())
importances = rf.feature_importances_

std = np.std([tree.feature_importances_ for tree in rf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), np.array(used_cols)[indices], rotation = 90)

plt.xlim([-1, X.shape[1]])

plt.show()
rf_imp_feat = ['total day charge','total day minutes','total eve charge']
#smote to affect imbalances



smote = SMOTE()

X_resampled, y_resampled = smote.fit_resample(X, y)



N = 5



rf = RandomForestClassifier(max_depth = N, n_estimators=100, n_jobs=-1, 

                            random_state=42,criterion = 'entropy', 

                            max_leaf_nodes = 2**N-1)

encoder = ForestEncoder(rf)

clf = LogisticRegression(class_weight='balanced')

pipe = make_pipeline(encoder, clf)

pipe.fit(X_resampled, y_resampled)



scores = cross_val_score(rf, X_resampled, y_resampled, cv=5, scoring='roc_auc')

print(scores.mean(), "+/-", scores.std())
#create train test split for resampled data



X_train, X_test, y_train, y_test = train_test_split(

    X_resampled, y_resampled, test_size=0.2, random_state=1, stratify=y_resampled)
grid_p = {"n_estimators": [20, 50, 100],

          "criterion": ["gini", "entropy"],

          "max_features": ['sqrt', 'log2', 0.2],

          "max_depth": [4, 6, 10],

          "min_samples_split": [2, 5, 10],

          "min_samples_leaf": [1, 5, 10]}



grid_search = GridSearchCV(rf, grid_p, n_jobs=-1, cv=5, scoring='roc_auc')

grid_search.fit(X_train, y_train)
grid_search.best_score_
grid_search.best_params_
rf = RandomForestClassifier(criterion='entropy',

 max_depth=10,

 max_features='sqrt',

 min_samples_leaf=1,

 min_samples_split=5,

 n_estimators=20)

encoder = ForestEncoder(rf)

clf = LogisticRegression(class_weight='balanced')

pipe = make_pipeline(encoder, clf)

pipe.fit(X_resampled, y_resampled)



scores = cross_val_score(rf, X_resampled, y_resampled, cv=5, scoring='roc_auc')

print(scores.mean(), "+/-", scores.std())
rf = RandomForestClassifier(criterion='entropy',

 max_depth=10,

 max_features='sqrt',

 min_samples_leaf=5,

 min_samples_split=2,

 n_estimators=100)

encoder = ForestEncoder(rf)

clf = LogisticRegression(class_weight='balanced')

pipe = make_pipeline(encoder, clf)

pipe.fit(X_resampled, y_resampled)



scores = cross_val_score(rf, X_resampled, y_resampled, cv=5, scoring='roc_auc')

print(scores.mean(), "+/-", scores.std())