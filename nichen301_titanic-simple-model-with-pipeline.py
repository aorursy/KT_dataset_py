import numpy as np

import pandas as pd



from matplotlib import pyplot as plt

plt.style.use("ggplot")

import seaborn as sns



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, RandomizedSearchCV



import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv("../input/train.csv")

n_train = len(train) 

X_train = train.drop(['PassengerId', 'Survived'], axis=1)

y_train = train.Survived



test = pd.read_csv("../input/test.csv")

X_test = test.drop('PassengerId', axis=1)



X_full = pd.concat([X_train, X_test], ignore_index=True)
# train.head()



# sns.countplot(x='Sex', hue='Survived', data=train)

# plt.gca().set_title('hh')

# plt.show()



# sns.countplot(x='Pclass', hue='Survived', data=train)



# train.groupby(['Pclass', 'Sex'])['Survived'].mean()

# # train.groupby(['Pclass', 'Sex'])['Survived'].mean().plot.bar()



# sns.catplot(x='Pclass', y='Survived', hue='Sex', data=train)



# sns.violinplot(x='Pclass', y='Age', hue='Survived', data=train, split=True)



# train.head()
na_info_train = train.isnull().sum()

na_info_test = test.isnull().sum()



na_info = pd.concat([na_info_train, na_info_test], axis=1)

na_info.columns = ['train', 'test']



na_info['total'] = na_info.train + na_info.test

na_info['total_pct'] = na_info.total / (len(train) + len(test))

na_info['dtype'] = train.dtypes[na_info.index]



na_info.sort_values(['total', 'test', 'train'], ascending=False, inplace=True)

na_info = na_info[na_info.total > 0]



na_info
class AwesomeImputer(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.fill_with_None = ["Cabin", "Embarked"]

        self.fill_with_mean = ["Age"]

        

    def fit(self, X, y=None):

        self.imp_mean = SimpleImputer(strategy="mean")

        self.imp_mean.fit(X[self.fill_with_mean])

        return self

    

    def transform(self, X):

        X[self.fill_with_None] = X[self.fill_with_None].fillna("None")

        X[self.fill_with_mean] = self.imp_mean.transform(X[self.fill_with_mean])

        X["Fare"] = X.groupby("Pclass")["Fare"].transform(lambda series: series.fillna(series.mean()))

        assert(X.isnull().sum().sum() == 0)

        return X
class Dummies(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        self._X = pd.get_dummies(X).head(1)

        return self



    def transform(self, X):

        X = pd.get_dummies(X)

        _, X = self._X.align(X, axis=1, join='left', fill_value=0)

        return X
class AwesomeScaler(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.scaler = RobustScaler()

    

    def fit(self, X, y=None):

        self.scaler.fit(X)

        return self

    

    def transform(self, X):

        # Apply scaler while preserving pd.DataFrame format. Otherwise it casts X to np.ndarray

        X[X.columns] = self.scaler.transform(X)

        return X
preprosessing = [

    AwesomeImputer(),

    Dummies(),

    AwesomeScaler(),

]

pipe_prep = make_pipeline(*preprosessing)
X_full_prep = pipe_prep.fit_transform(X_full)

X_train_prep = X_full_prep[:n_train]

X_test_prep = X_full_prep[n_train:]
from scipy.stats import randint, uniform, expon



def random_search(model, param_dist, X, y, n_iter=10):

    assert(len(param_dist) <= 6)

    rand_search = RandomizedSearchCV(model, param_dist, scoring="accuracy", cv=5, n_iter=n_iter, verbose=1)

    rand_search.fit(X, y)

    print(rand_search.best_score_, rand_search.best_params_)

    f, axes = plt.subplots(len(param_dist)//2 + 1, 2)

    for idx, param in enumerate(param_dist):

        ax = axes.flatten()[idx]

        sns.scatterplot(rand_search.cv_results_['param_{}'.format(param)], \

                        rand_search.cv_results_['mean_test_score'], ax=ax)

        ax.set_xlabel(param)

        ax.set_ylabel('accuracy')



class ExpoUni(object):

    def __init__(self, loc, end):

        self.loc = loc

        self.scale = end - loc

        self.uni = uniform(self.loc, self.scale)

        

    def rvs(self, size=None, random_state=None):

        uniform_samples = self.uni.rvs(size=size, random_state=random_state)

        return np.power(10, uniform_samples)
# random_search(SVC(gamma=0.03), {'C': uniform(loc=10, scale=50)}, X=X_train_prep, y=y_train, n_iter=20)
# random_search(SVC(C=10), {'gamma': uniform(loc=0.02, scale=0.06)}, X=X_train_prep, y=y_train)
model_subm = SVC(C=20, gamma=0.02)
# cross_val_score(model_subm, X=X_train_prep, y=y_train, scoring='accuracy', verbose=1)
model_subm.fit(X=X_train_prep, y=y_train)

y_pred = model_subm.predict(X_test_prep)

result = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred})

result.to_csv("submission.csv", index=False)