import numpy as np
import pandas as pd 
import matplotlib as plt
import seaborn as sns


df = pd.read_csv('../input/train.csv')

df.info()
corr = df.corr()
sns.heatmap(corr, cmap=sns.color_palette("coolwarm", 8))
df = df.dropna(axis=0, how='any', subset=['Embarked'])

X_train_ = df.drop('Survived', axis=1)
y_train = df['Survived']

X_train_.info()
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class FamilySizeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        familySize = X[:, 2] + X[:, 3] + 1
        isAlone = familySize == 1
        return np.c_[X, familySize, isAlone]
    
class TitleTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        possible_title = ['Mr', 'Master', 'Miss', 'Mrs']
        for title in possible_title:
            temp = np.char.find(X[:,0].astype(str), title) >= 0
            X = np.c_[X, temp]
        return X[:,1:]
    
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values

class LabelBinarizerPipelineFriendly(MultiLabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)
    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Imputer, StandardScaler

num_attribs = ["Age", "Pclass", "SibSp","Parch", "Fare"]
cat_attribs = ["Sex"]
name_attribs = ["Name"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('family', FamilySizeTransformer()),
    ('imputer', Imputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', LabelBinarizerPipelineFriendly())
])

name_pipeline = Pipeline([
    ('selector', DataFrameSelector(name_attribs)),
    ('title', TitleTransformer())
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
    ('name_pipeline', name_pipeline)
])
print(X_train_.loc[7])
X_train = full_pipeline.fit_transform(X_train_)
print(X_train[7])
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

param_dist = {"max_depth": [3, None],
              "max_features": [2, 3, 9, 13],
              "min_samples_split": [2, 3, 9, 27, 81],
              "min_samples_leaf": [2, 3, 9, 27, 81],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions = param_dist, n_iter = 100)
random_search.fit(X_train, y_train)


forest_clf = random_search.best_estimator_
train_predict = forest_clf.predict(X_train)
from sklearn.metrics import precision_score, recall_score
## use cross_val_score() with K = 3 to measure model performance 
from sklearn.model_selection import cross_val_score

print("accuracy"  + str(cross_val_score(forest_clf, X_train, y_train, cv=3, scoring='accuracy')))
print("precision: " + str(precision_score(y_train, train_predict)))
print("recall: " + str(recall_score(y_train, train_predict)))
df_test = pd.read_csv('../input/test.csv')
X_test = full_pipeline.fit_transform(df_test)
predictions = forest_clf.predict(X_test)

submission = pd.DataFrame({ 'PassengerId': df_test['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission3.csv", index=False)
