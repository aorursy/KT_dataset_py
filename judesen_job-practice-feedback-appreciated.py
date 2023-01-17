# Data processing
import numpy as np
import pandas as pd

# Data visualization
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
%matplotlib inline

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Eliminating unnecessary sklearn warnings
import warnings
warnings.filterwarnings('ignore')
# Create table for missing data analysis
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
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
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
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
# Plot validation curve
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid() 
    plt.xscale('log')
    plt.legend(loc='best') 
    plt.xlabel('Parameter') 
    plt.ylabel('Score') 
    plt.ylim(ylim)
df = pd.read_csv('../input/train.csv') # Load the data and save as df
df_raw = df.copy() # Save the data as raw format just in case
df.head()
df.describe()
# Draw a data table showing the missing data as percentage of total
draw_missing_data_table(df)
df.dtypes
# Define date as date.
df['date'] = pd.to_datetime(df['date'])
df.drop('date', axis=1, inplace=True)
# Define store and item as categorical
df['store'] = pd.Categorical(df['store'])
df['item'] = pd.Categorical(df['item'])
df.head()
# Transform categorical variables into dummy variables
df = pd.get_dummies(df, drop_first=True)  # To avoid dummy trap
df.head()
# Create data set to train data imputation methods
X = df[df.loc[:, df.columns != 'sales'].columns]
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)
# Debug
print('Inputs: \n', X_train.head())
print('Outputs: \n', y_train.head())
print(X.head())
# Fit linear regression
sgdreg = SGDRegressor()
sgdreg.fit(X_train, y_train)
# Model performance
scores = cross_val_score(sgdreg, X_train, y_train, cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# Plot learning curves
title = "Learning Curves (Linear Regression)"
cv = 10
plot_learning_curve(estimator=sgdreg, title=title, X=X_train, y=y_train, cv=cv, n_jobs=1);
# Plot validation curve
title = 'Validation Curve (Linear Regression)'
param_name = 'alpha'
param_range = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3] 
cv = 10
plot_validation_curve(estimator=sgdreg, title=title, X=X_train, y=y_train, param_name=param_name, param_range=param_range);
# Restart data set
df = df_raw.copy()
df.head()
df.dtypes
# Define date as date.
df['date'] = pd.to_datetime(df['date'])
# Define store and item as categorical
df['store'] = pd.Categorical(df['store'])
df['item'] = pd.Categorical(df['item'])
df.dtypes
plt.rcParams['figure.figsize']=(20,5)
sns.barplot(df['item'], df['sales'])
plt.rcParams['figure.figsize']=(10,5)
sns.barplot(df['store'], df['sales'])
g = sns.FacetGrid(df, col="item", col_wrap=5)
g = g.map(plt.scatter, "date", "sales", marker="o", s=1, alpha=.5)
df['month'] = df.date.dt.month
df['month'] = pd.Categorical(df['month'])
df.head()
sns.barplot(df['month'],df['sales']);
df.dtypes
# Add SMA to dataframe
sma = df.groupby(['item', 'store'], as_index=False).apply(lambda x: x['sales'].rolling(90).mean().shift(365))
df['sma'] = sma.reset_index(level=0, drop=True)

# Add EMA to dataframe
ema = df.groupby(['item', 'store'], as_index=False).apply(lambda x: x['sales'].ewm(span=90, adjust=False).mean().shift(365))
df['ema'] = ema.reset_index(level=0, drop=True)

df.head()
df_latest_year_one_item = df.loc[(df['date'] > '2014-12-31') & (df['item'] == 1) & (df['store'] == 10)]
plt.figure(figsize=(20,5))
plt.plot(df_latest_year_one_item[['sales','sma', 'ema']])
plt.legend(['sales','sma','ema'])
plt.show()
df.isna()['sma'].sum()
df.dropna(inplace=True)
df.isna()['sma'].sum()
# Drop date
df.drop('date', axis=1, inplace=True)
# Transform categorical variables into dummy variables
df = pd.get_dummies(df, drop_first=True)  # To avoid dummy trap
# Create data set to train data imputation methods
X = df[df.loc[:, df.columns != 'sales'].columns]
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)
# Debug
print('Inputs: \n', X_train.head())
print('Outputs: \n', y_train.head())
# Scale the features
columns_to_scale = ['sma', 'ema']
mean = X_train[columns_to_scale].mean()
std = X_train[columns_to_scale].std()
X_train[columns_to_scale] = (X_train[columns_to_scale] - mean) / std

# Fit linear regression
sgdreg = SGDRegressor(alpha=0.0001)
sgdreg.fit(X_train, y_train)
# Model performance
scores = cross_val_score(sgdreg, X_train, y_train, cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# Plot learning curves
title = "Learning Curves (Linear Regression)"
cv = 10
plot_learning_curve(estimator=sgdreg, title=title, X=X_train, y=y_train, cv=cv, n_jobs=1);
# Plot validation curve
title = 'Validation Curve (Linear Regression)'
param_name = 'alpha'
param_range = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3] 
cv = 10
plot_validation_curve(estimator=sgdreg, title=title, X=X_train, y=y_train, param_name=param_name, param_range=param_range);
# Restart data set
df = df_raw.copy()
df_test_raw = pd.read_csv('../input/test.csv') # Load the test data and save as df_test
df_test = df_test_raw.copy()

combine = [df, df_test]
# Define date as date.
for dataset in combine:
    dataset['date'] = pd.to_datetime(dataset['date'])
# Define store and item as categorical
for dataset in combine:
    dataset['store'] = pd.Categorical(dataset['store'])
    dataset['item'] = pd.Categorical(dataset['item'])
# Define month as categorical
for dataset in combine:
    dataset['month'] = dataset.date.dt.month
    dataset['month'] = pd.Categorical(dataset['month'])

df_test.dtypes
# Add SMA to dataframe
sma = df.groupby(['item', 'store'], as_index=False).apply(lambda x: x['sales'].rolling(90).mean())
df['sma'] = sma.reset_index(level=0, drop=True)

# Add EMA to dataframe
ema = df.groupby(['item', 'store'], as_index=False).apply(lambda x: x['sales'].ewm(span=90, adjust=False).mean())
df['ema'] = ema.reset_index(level=0, drop=True)
# Adding last 3 months of sma and ema from training to test data
sma_test = df.loc[(df['date'] < '2017-04-01') & (df['date'] >= '2017-01-01')]['sma'].reset_index(drop=True)
df_test['sma'] = sma_test
ema_test = df.loc[(df['date'] < '2017-04-01') & (df['date'] >= '2017-01-01')]['ema'].reset_index(drop=True)
df_test['ema'] = ema_test

# Shifting SMA and EMA on training data
df['sma'] = df['sma'].shift(365)
df['ema'] = df['ema'].shift(365)

df_test.head()
# Drop date and id
df.drop('date', axis=1, inplace=True)
df.dropna(inplace=True)
df_test.drop('date', axis=1, inplace=True)
df_test.drop('id', axis=1, inplace=True)
# Transform categorical variables into dummy variables
df = pd.get_dummies(df, drop_first=True)  # To avoid dummy trap
df_test = pd.get_dummies(df_test, drop_first=True)  # To avoid dummy trap

# Add month 4-12 to df_test
df_test = df_test.join(pd.DataFrame(
    {
        'month_4': 0,
        'month_5': 0,
        'month_6': 0,
        'month_7': 0,
        'month_8': 0,
        'month_9': 0,
        'month_10': 0,
        'month_11': 0,
        'month_12': 0
    }, index=df_test.index
))
df_test.head()
# Prepare data for model
X_train = df[df.loc[:, df.columns != 'sales'].columns]
y_train = df['sales']
X_test = df_test

# Scale the features
columns_to_scale = ['sma', 'ema']
mean = X_train[columns_to_scale].mean()
std = X_train[columns_to_scale].std()
X_train[columns_to_scale] = (X_train[columns_to_scale] - mean) / std
X_test[columns_to_scale] = (X_test[columns_to_scale] - mean) / std
# Run SGD Regression
sgdreg = SGDRegressor(alpha=0.0001)
sgdreg.fit(X_train, y_train)

# Get prediction
prediction = sgdreg.predict(X_test)
# Add to submission
submission = pd.DataFrame({
        "id": df_test_raw['id'],
        "sales": prediction
})
# Quick look at the submission
submission.head()
# Save submission
submission.to_csv('submission.csv',index=False)