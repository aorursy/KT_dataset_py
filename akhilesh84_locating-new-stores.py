# Import dependencies

# Numeric arithematic
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scientific compulation and machine learning
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, learning_curve, cross_val_score, LeaveOneOut
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
raw_data = pd.read_excel(io="../input/Module_3_excercises.xls", sheet_name="3.Retail", usecols="B:H")
raw_data.head(5)
raw_data.describe()
ax = sns.boxplot(data=raw_data, y="income")
ax.set_xlabel("Income");
raw_data[raw_data.income > 55000]
columns_to_dummify = ["promotion"]

df = raw_data.iloc[:].copy()

for column in columns_to_dummify:
    if column in df.columns.values:
        temp = pd.get_dummies(df[column])
        temp.columns = [column + '_' + str(i) for i in temp.columns.values]
        temp.set_index(df.index.values)
        df = pd.concat([df, temp], axis=1)
        df.drop([column], axis=1, inplace=True)
df.head(5)
sns.pairplot(df,
             x_vars=["advertising", "miles", "sqfeet", "% owners", "income", "sales"],
             y_vars=["advertising", "miles", "sqfeet", "% owners", "income", "sales"]
            );
X = df.iloc[:, [0,1,2,3,4,6,7, 8]]
y = df.iloc[:, [5]]
X_Scaler = MinMaxScaler(feature_range=(0,1))
y_Scaler = MinMaxScaler(feature_range=(0,1))

X_scaled = DataFrame(X_Scaler.fit_transform(X), columns=X.columns.values)
y_scaled = DataFrame(y_Scaler.fit_transform(y), columns=y.columns.values)
X_scaled.head(5)
y_scaled.head(5)
# courtesy of code at http://scikit-learn.org/stable/modules/learning_curve.html#learning-curve
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

random_state = 45749
kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
model = Ridge(alpha = .1)
model.fit(X_scaled, y_scaled)
results = cross_val_score(model, X_scaled, y_scaled, cv = kfold)
print("{0} => Accuracy: {1:.2f}%, Standard Deviation: (+/-){2:.2f}%, R Square Value: {3:.2f}%".format("Ridge Regression", results.mean() * 100, results.std() * 100, model.score(X_scaled, y_scaled)*100))
plot_learning_curve(model, "Ridge Regression", X_scaled, y_scaled, (0, 1), cv=kfold)
plt.show()
