import re



import numpy as np

import pandas as pd

import matplotlib as plt

import seaborn as sns



from sklearn.base import TransformerMixin

from sklearn.preprocessing import FunctionTransformer, StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.pipeline import make_union, make_pipeline



sns.set()
%matplotlib inline
df_train = pd.read_csv("../input/train.csv", na_values="NaN", index_col=0)

df_test = pd.read_csv("../input/test.csv", na_values="NaN", index_col=0)
survived = df_train['Survived']

df_train.drop(labels=['Survived'], axis=1, inplace=True)

df_train['Survived'] = survived
df_train.head()
df_test.head()
p = df_train.groupby("Pclass").size().plot(kind="bar")

p.set_xlabel("Class")

p.set_ylabel("Count")

p
df_train.pivot_table("Name", "Pclass", "Survived", "count").plot(kind="bar", stacked=True)
len(df_train[df_train.Pclass.isnull()])
len(df_train[df_train.Name.isnull()])
df_train.pivot_table("Name", "Sex", "Survived", "count").plot(kind="bar", stacked=True)
len(df_train[df_train.Sex.isnull()])
df_train.pivot_table("Name", "Age", "Survived", "count").plot(kind="bar", stacked=True, figsize=(20, 10))
len(df_train[df_train.Age.isnull()])
df_train.pivot_table("Name", "SibSp", "Survived", "count").plot(kind="bar", stacked=True)
len(df_train[df_train.SibSp.isnull()])
df_train.pivot_table("Name", "Parch", "Survived", "count").plot(kind="bar", stacked=True)
len(df_train[df_train.Parch.isnull()])
len(df_train[df_train.Ticket.apply(lambda x: len(x.split())) > 1])
len(df_train[df_train.Ticket.isnull()])
df_train.Fare.corr(-df_train.Pclass)
len(df_train[df_train.Fare.isnull()])
len(df_train[df_train.Fare == 0.0])
len(df_train[df_train.Cabin.isnull()])
sum(pd.isnull(df_train.Cabin)) / len(df_train)
df_train.pivot_table("Name", "Embarked", "Survived", "count").plot(kind="bar", stacked=True)
_, axes = plt.pyplot.subplots(ncols=2, figsize=(10, 5))

df_train.pivot_table("Name", "Embarked", "Sex", "count").plot(kind="bar", stacked=True, ax=axes[0])

df_train.pivot_table("Name", "Embarked", "Pclass", "count").plot(kind="bar", stacked=True, ax=axes[1])
class FeatureExtractor(TransformerMixin):



    def __init__(self, new_feature_name, extractor_function):

        self.new_feature_name = new_feature_name

        self.extractor_function = extractor_function

    

    def fit(self, X, y=None):

        return self



    def transform(self, X, y=None):

        X[self.new_feature_name] = self.extractor_function(X)

        return X
class MeanByCategoryImputer(TransformerMixin):



    def __init__(self, group_key, mean_key, nan_value=None):

        self.group_key = group_key

        self.mean_key = mean_key

        self.nan_value = nan_value

    

    def fit(self, X, y=None):

        self.means_by_cat = X.groupby(self.group_key).mean()[self.mean_key].to_dict()

        return self



    def transform(self, X, y=None):

        if self.nan_value:

            X[X[self.mean_key] == self.nan_value] = np.nan

        X[self.mean_key] = X[self.mean_key].fillna(X[self.group_key].map(self.means_by_cat))

        if sum(X[self.mean_key].isnull()) > 0: # we have a 1-member group

            X[self.mean_key] = X[self.mean_key].fillna(X[self.mean_key].mean())

        return X[[self.mean_key]]
class LabelEncoderPipelineFriendly(LabelEncoder):

    

    def fit(self, X, y=None):

        """this would allow us to fit the model based on the X input."""

        super(LabelEncoderPipelineFriendly, self).fit(X)

        

    def transform(self, X, y=None):

        return super(LabelEncoderPipelineFriendly, self).transform(X).reshape(-1, 1)



    def fit_transform(self, X, y=None):

        return super(LabelEncoderPipelineFriendly, self).fit(X).transform(X).reshape(-1, 1)
class FeaturesSum(TransformerMixin):

    

    def fit(self, X, y=None):

        return self

        

    def transform(self, X, y=None):

        return np.sum(X.astype(np.float64), axis=1).values.reshape(-1, 1)



    def fit_transform(self, X, y=None):

        return self.transform(X)
def prepare_pipeline():

    def get_age_col(X):

        return X.copy()[["Age", "Name"]] #  mutation ahead

    

    def get_title(X):

        return X[["Name"]].apply(lambda x: re.match(".*\, ((the )?\S*)\. .*", x.Name).groups()[0], axis=1)

    

    def get_pclass_col(X):

        return X[["Pclass"]]

    

    def get_sex_col(X):

        return X["Sex"] #  LabelEncoder expects 1d array

    

    def get_sum_col(X):

        return X[["SibSp", "Parch"]]

    

    def get_ticket_prefix(X):

        def extract_prefix(x):

            match = re.match("(.*) .*", x.Ticket.replace(".", ""))

            if match or x.Ticket == "LINE":

                return 1

            return 0

        return X[["Ticket"]].apply(extract_prefix, axis=1).values.reshape(-1, 1)

    

    def get_cabin(X):

        return X["Cabin"].isnull().astype(int) #  LabelEncoder expects 1d array

    

    pipeline = make_union(*[

        make_pipeline(FunctionTransformer(get_pclass_col, validate=False), OneHotEncoder(sparse=False)),

        make_pipeline(FunctionTransformer(get_sex_col, validate=False), LabelEncoderPipelineFriendly()),

        make_pipeline(FunctionTransformer(get_age_col, validate=False),

                      FeatureExtractor("Title", get_title), 

                      MeanByCategoryImputer("Title", "Age"),

                      StandardScaler()),

        make_pipeline(FunctionTransformer(get_sum_col, validate=False), FeaturesSum(), StandardScaler()),

        make_pipeline(FunctionTransformer(get_ticket_prefix, validate=False), OneHotEncoder(sparse=False)),

        make_pipeline(MeanByCategoryImputer("Pclass", "Fare", 0.0), StandardScaler()),

        make_pipeline(FunctionTransformer(get_cabin, validate=False), LabelEncoderPipelineFriendly())

        

    ])

    return pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score



from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC



models = [

    (KNeighborsClassifier, {"n_neighbors": list(range(1, 21))}),

    (LogisticRegression, {"penalty": ["l1", "l2"], 

                          "C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0] + list(range(10, 101, 10)),

                          "max_iter": list(range(100, 501, 100)),

                          "random_state": [0]}),

    (SVC, {"C": [1, 10, 100, 1000], 

           "gamma": [0.1, 0.01, 0.001, 0.0001], 

           "kernel": ["rbf", "linear"],

           "random_state": [0]})

]
x = prepare_pipeline().fit_transform(df_train)

y = df_train.Survived
best_models = []



for model_class, params in models:

    np.random.seed = 0

    gs = GridSearchCV(model_class(), params, scoring="accuracy", cv=10, n_jobs=8)

    gs.fit(x, y)

    best_models.append((model_class, gs.best_estimator_, gs.best_params_, gs.best_score_))
scores = []

for m in best_models:

    model_class, _, params, _ = m

    estimator = model_class(**params)

    local_scores = cross_val_score(estimator, x, y, cv=20, scoring="accuracy")

    scores.append((estimator, local_scores.mean(), local_scores.std()))
best_model = max(scores, key=lambda x: x[1])[0]
best_model.fit(x, y)
test = prepare_pipeline().fit_transform(df_test)
prediction = best_model.predict(test)
result = pd.DataFrame({"PassengerId": df_test.index, "Survived": prediction})
result.to_csv("submission.csv", sep=",", index=False)