# imports
from xgboost import XGBRegressor
from scipy.stats import skew  
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
# Load train and test data from files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv('../input/test.csv')

features_to_drop = set()
# describe all the numerical variables
train.select_dtypes(exclude = ["object"]).describe()
#descriptive statistics summary
train['SalePrice'].describe()
#histogram
sns.distplot(train['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())
#scatter plot grlivarea/saleprice
train.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice
f, ax = plt.subplots(figsize=(12, 8))
fig = sns.boxplot(x=train['YearBuilt'], y=train['SalePrice'])
# fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();

corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
features_to_drop |= set(percent[percent > 0.15].index)
print("To drop: {}".format(features_to_drop))
# Separate target values from the training set
y = train.SalePrice
train = train.drop("SalePrice", 1)

categorical_features = train.select_dtypes(include = ["object"]).columns
numerical_features = train.select_dtypes(exclude = ["object"]).columns

X = train.copy()
class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, drop=[]):
        self.drop = drop

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X[list(set(X.columns) - set(self.drop))]
# try    
before = train.columns
t = FeatureDropper(features_to_drop)
t.fit(train)
train = t.transform(train)
print("To drop: {} \nDropped: {}".format(features_to_drop, set(before) - set(train.columns)))
# this transformer handles missing values 
class FillNaTransformer(BaseEstimator, TransformerMixin):    
    def __init__(self, columns = None):
        self.columns = columns
        
    def fit(self, X, y=None, **fit_params):
        if self.columns is None:
            self.columns = X.select_dtypes(exclude = ["object"]).columns
        self.train_median = X[self.columns].median()
        return self

    def transform(self, X):
        X[self.columns] = X[self.columns].fillna(self.train_median) 
        return X
#     check
#  print nan counts before and after
print(train.select_dtypes(exclude = ["object"]).isnull().sum().sum())
t = FillNaTransformer()
t.fit(train)
train = t.transform(train)
print(train.select_dtypes(exclude = ["object"]).isnull().sum().sum())
# this transformer applies log to skewed features
class FixSkewnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        if self.columns is None:
            self.columns = X.select_dtypes(exclude = ["object"]).columns 
        skewness = X[self.columns].apply(lambda x: skew(x))
        skewness = skewness[abs(skewness)>0.5]
        self.skew_features = skewness.index
        return self

    def transform(self, X):
        X[self.skew_features] = np.log1p(X[self.skew_features])
        return X
tskew = FixSkewnessTransformer()
tskew.fit(train)
train = tskew.transform(train)
class OneHotEncoder(BaseEstimator, TransformerMixin):
    # numerical - numerical columns to be treated as categorical
    # columns - columns to use (if None then all categorical variables are included)
    def __init__(self, columns=None, numerical=[]):
        self.numerical = numerical
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        # if none specified – get all non numerical columns
        if self.columns == None:
            self.columns = X.select_dtypes(include = ["object"]).columns.tolist()
        self.columns += self.numerical
        # get all possible column values to filter not seen values
        self.allowed_columns = [ "{}_{}".format(column, val) for column in self.columns for val in X[column].unique() ]
        return self

    def transform(self, X, y=None, **fit_params):
        # cast numerical columns to strings 
        for col in X[self.columns].select_dtypes(exclude = ["object"]).columns:
            X[col] = X[col].astype('str')
        one_hots = pd.get_dummies(X[self.columns], prefix=self.columns)
        missing_cols = set(self.allowed_columns) - set(one_hots.columns)
        for c in missing_cols:
            one_hots[c] = 0
        return pd.concat([X.drop(self.columns, axis=1), one_hots.filter(self.allowed_columns)], axis=1)
#     check
print(train.columns)
to = OneHotEncoder()
to.fit(train)
train = to.transform(train)
print(train.columns)
print(train.select_dtypes(include = ["object"]).columns)



pipe = Pipeline([
    ('drop', FeatureDropper(features_to_drop)),
    ('fillna', FillNaTransformer()),
#     ('skew', FixSkewnessTransformer()),
    ('onehot', OneHotEncoder()),
#     ('scale', StandardScaler()),
    ('regressor', XGBRegressor())
])
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.2, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="neg_mean_squared_error")
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = -np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = -np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
plot_learning_curve(pipe, 'learning curve', X.copy(), y, cv=5, n_jobs=4)
from sklearn.model_selection import GridSearchCV

# up to 3 grid dimensions (choose max from the last one)
def plot_grid(grid, k_options, l_options, m_options_length, k_label="k options"):
    mean_scores = np.array(-grid.cv_results_['mean_test_score'])
    # scores are in the order of param_grid iteration, which is alphabetical
    mean_scores = mean_scores.reshape(m_options_length, -1, len(k_options))
    # select score for best C
    mean_scores = mean_scores.max(axis=0)
    bar_offsets = (np.arange(len(k_options)) *
                   (len(l_options) + 1) + .5)

    plt.figure()
    COLORS = 'bgrcmyk'
    for i, (label, l_scores) in enumerate(zip(l_options, mean_scores)):
        plt.bar(bar_offsets + i, l_scores, label=label, color=COLORS[i])

    plt.title("Search for best hyperparams")
    plt.xlabel(k_label)
    plt.xticks(bar_offsets + len(l_options) / 2, k_options)
    plt.ylabel('Mean score')
    plt.legend(loc='best')
    plt.show()

param_grid = [
    {
        'regressor__max_depth': [2,3,4],
        'regressor': [XGBRegressor()]
    },
    {
        'regressor__C': [1, 10, 100],
        'regressor': [SVR()]
    },
    {
        'regressor__n_estimators': [5,10,20],
        'regressor': [RandomForestRegressor()]
    },
]
regressor_labels = ['xgboost', 'svr', 'random forrest']

grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid, scoring='neg_mean_squared_error')
# grid.fit(X.copy(), y)

# plot_grid(grid, [1,2,3], regressor_labels, 1)
# grid.best_params_
# pipe.set_params(**grid.best_params_)
def rmsle(y, y_pred, **kwargs):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y_pred), 2)))

scorer = make_scorer( rmsle )
score = cross_val_score(pipe, X.copy(), y, cv=5, scoring=scorer, n_jobs=4).mean()
print("Score: {}".format(score))

pipe.fit(X,y)
predicted_prices = pipe.predict(test)
print(predicted_prices)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)