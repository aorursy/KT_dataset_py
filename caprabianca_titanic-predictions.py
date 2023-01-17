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



df = pd.read_csv('../input/train.csv', index_col = [0])

df1 = pd.read_csv('../input/test.csv', index_col = [0])

df2 = pd.read_csv('../input/gender_submission.csv')
df.head()
import seaborn as sns



sns.heatmap(df.corr(), cmap='Greens')
df.corr()
%matplotlib inline

import matplotlib.pyplot as plt

df.hist(bins=50, figsize=(20,15))

plt.show()
#from sklearn.model_selection import StratifiedShuffleSplit



#split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



#for train_index, test_index in split.split(df, df['Pclass']):

#    strat_train_set = df.iloc[train_index]

#    strat_test_set = df.iloc[test_index]
#df_labels = strat_train_set["Survived"].copy()

df_labels = df["Survived"].copy()



#df = strat_train_set.drop(["Survived", "Name", "Ticket", "Cabin"] , axis=1)

df = df.drop(["Survived", "Name", "Ticket", "Cabin"] , axis=1)
from sklearn.impute import SimpleImputer



imputer_cat = SimpleImputer(strategy='most_frequent')

imputer_num = SimpleImputer(strategy='median')

#Median only works with numerical values.



df_num = df.drop(["Sex", "Embarked"], axis=1)

df_cat = df[["Sex", "Embarked"]]



imputer_cat.fit(df_cat)

imputer_num.fit(df_num)



num = imputer_num.transform(df_num)

cat = imputer_cat.transform(df_cat)



df_num_tr = pd.DataFrame(num, columns=df_num.columns)

df_cat_tr = pd.DataFrame(cat, columns=df_cat.columns)
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(categories='auto')



df_cat_1hot = encoder.fit_transform(df_cat_tr.values.reshape(-1,1))
from sklearn.base import BaseEstimator, TransformerMixin



siblings, parents = 2,3



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, sub_tot = True):

        self.sub_tot = sub_tot

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        if self.sub_tot:

            tot = X[:, siblings] + X[:, parents]

            return np.c_[X[:, [0,1,4]], tot]

        else:

            return np.c_[X]
attr_adder = CombinedAttributesAdder()

housing_extra_attribs = attr_adder.transform(df_num_tr.values)
from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values
from sklearn.pipeline import FeatureUnion

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_attribs = list(df_num)

cat_attribs = list(df_cat)



num_pipeline = Pipeline([

        ('selector', DataFrameSelector(num_attribs)),

        ('imputer', SimpleImputer(strategy='median')), 

        ('attribs_adder', CombinedAttributesAdder()),

        ('std_scaler', StandardScaler()),

])



cat_pipeline = Pipeline([

        ('selector', DataFrameSelector(cat_attribs)),

        ('imputer', SimpleImputer(strategy='most_frequent')), 

        ('label_binarizer', OneHotEncoder(categories='auto')),

])



full_pipeline = FeatureUnion(transformer_list=[

    ('num_pipeline', num_pipeline),

    ('cat_pipeline', cat_pipeline),

])
df_prepared = full_pipeline.fit_transform(df).toarray()
from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm



sgd_clf = SGDClassifier(max_iter=5, tol=None, random_state=42)

rfc_clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

svm_clf = svm.SVC()

lin_clf = svm.LinearSVC()



sgd_clf.fit(df_prepared, df_labels)

rfc_clf.fit(df_prepared, df_labels)

lin_clf.fit(df_prepared, df_labels)

svm_clf.fit(df_prepared, df_labels)
from sklearn.model_selection import cross_val_score



cross_val_score(rfc_clf, df_prepared, df_labels, cv=5, scoring='accuracy')
from sklearn.model_selection import cross_val_score



cross_val_score(lin_clf, df_prepared, df_labels, cv=5, scoring='accuracy')
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix



y_train_pred = cross_val_predict(sgd_clf, df_prepared, df_labels, cv=5)

confusion_matrix(df_labels, y_train_pred)
from sklearn.model_selection import cross_val_score



cross_val_score(svm_clf, df_prepared, df_labels, cv=5, scoring='accuracy')
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

from scipy import stats



param_dist = {"gamma": stats.uniform(0.1, 100),

              "C": stats.uniform(0.1, 1000),

              "degree": randint(low=0, high=6)

             }



random_search = RandomizedSearchCV(svm_clf, param_distributions=param_dist,

                                   n_iter=2000, cv=4, iid=False, scoring='accuracy')
#random_search.fit(df_prepared, df_labels)
def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")

            

report(random_search.cv_results_)
svm_clf = svm.SVC(gamma='scale')

svm_clf.fit(df_prepared, df_labels)

#cross_val_score(svm_clf, df_prepared, df_labels, cv=4, scoring='accuracy')
test = full_pipeline.transform(df1).toarray()

pred = svm_clf.predict(test)
submission = pd.read_csv('../input/gender_submission.csv')

submission['Survived'] = pred
submit = submission.to_csv ('sub.csv', index = None, header=True)