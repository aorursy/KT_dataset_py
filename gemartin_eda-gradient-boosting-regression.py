import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import learning_curve

from sklearn.model_selection import train_test_split, KFold, cross_val_score



%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print("train: {}".format(train.shape))

print("test: {}".format(test.shape))



train.head(10)
test.head(10)
train["y"].dtype
train["y"].describe()
plt.figure()

plt.boxplot(train["y"])

plt.ylabel("y")

plt.show()
# Get the IQR

q75, q25 = np.percentile(train["y"], [75, 25])

iqr = q75 - q25



minimum = q25 - (iqr * 1.5)

maximum = q75 + (iqr * 1.5)



print("Minimum = %.2f" % minimum)

print("Maximim = %.2f" % maximum)
plt.figure()

plt.boxplot(train["y"][(train["y"] >= minimum) & (train["y"] <= maximum)])

plt.ylabel("y")

plt.show()
plt.plot(train["ID"], train["y"])
periods = [10,20,50,100]



fig = plt.figure(figsize=(20,10))



for n in periods:

    col = "MA" + str(n)

    train[col] = train["y"].rolling(window=n).mean()



ax1 = fig.add_subplot(411)

ax1.plot(train["ID"], train["MA10"])



ax2 = fig.add_subplot(412)

ax2.plot(train["ID"], train["MA20"])



ax3 = fig.add_subplot(413)

ax3.plot(train["ID"], train["MA50"])



ax4 = fig.add_subplot(414)

ax4.plot(train["ID"], train["MA100"])
# The first n rows of the MAn columns have NA values. 

# We replace them by the average y

for col in ["MA10","MA20","MA50","MA100"]:

    train[col].fillna(train["y"].mean(), inplace=True)
cols = [c for c in train.columns if 'X' in c]

print('Number of features: {}'.format(len(cols)))



print('Feature types:')

train[cols].dtypes.value_counts()
counts = [[], [], []]

for c in cols:

    typ = train[c].dtype

    uniq = len(np.unique(train[c]))

    if uniq == 1: counts[0].append(c)

    elif uniq == 2 and typ == np.int64: counts[1].append(c)

    else: counts[2].append(c)



print('Constant features: {}\nBinary features: {}\nCategorical features: {}\n'.format(*[len(c) for c in counts]))



print('Constant features:', counts[0])

print('Categorical features:', counts[2])
cat_feat = counts[2]

train[cat_feat].head()
fig, ax = plt.subplots(8, 1, figsize=(30,40))

for c in cat_feat:

    axis = ax[cat_feat.index(c)]

    ax2 = axis.twinx()

    

    # plot with the outliers

    # sns.boxplot(x=train[c], y=train["y"], color="c", ax=ax[cat_feat.index(c)])

    

    # plot without the outiers

    sns.boxplot(x=train[c], y=train["y"][(train["y"] >= minimum) & (train["y"] <= maximum)], color="c", ax=axis)

    sns.countplot(x=train[c], alpha=0.3, color="c", ax=ax2)
binary_features = counts[1]

len(binary_features)
train["total"] = train[binary_features].sum(axis=1)
train.head()
print(train["total"].describe())

plt.boxplot(train["total"])

plt.show()
plt.scatter(train["total"], train["y"], alpha=0.1)
# Try again without the outliers

plt.scatter(train["total"][(train["y"] >= minimum) & (train["y"] <= maximum)], 

            train["y"][(train["y"] >= minimum) & (train["y"] <= maximum)], 

            alpha=0.1)
# Limit total values to IQR

plt.scatter(train["total"][(train["y"] >= minimum) & (train["y"] <= maximum) & (train["total"] >= 53) & (train["total"] <= 63)], 

            train["y"][(train["y"] >= minimum) & (train["y"] <= maximum) & (train["total"] >= 53) & (train["total"] <= 63)], 

            alpha=0.1)
test["total"] = test[binary_features].sum(axis=1)

train.drop(["MA10", "MA20", "MA50", "MA100"],axis=1, inplace=True)
def dummify(df, columns, drop=True):

    ''' add dummy variables columns to a dataframe

    

    parameters

    ----------

    df: dataframe

        the dataframe that need to be modified

        

    columns: list

        a list of column names for which we'll create dummy variables

        

    drop: boolean (default=True)

        True to drop the original column

            

    return

    ------

        a dataframe with extra dummy variables

    '''

    

    for column in columns:

        df_dummies = pd.get_dummies(df[column], prefix=column)

        df = pd.concat([df,df_dummies], axis=1)

        if drop == True:

            df.drop([column], inplace=True, axis=1)

    

    return df





def add_missing_dummy_columns(df, columns):

    ''' add missing dummy columns to a dataframe

    

        If a categorical feature in the test set doesn't

        have as many values than the same feature in the 

        train set, the two dataframes will not have the

        same number of dummy columns.

        

        This function add the dummy columns that are missing

        and fill them with zeros

        

    parameters

    ----------

    df: dataframe

        The dataframe with missing columns

    

    columns: list

        The complete list of dummy columns

    '''

    

    missing_cols = set(columns) - set(df.columns)

    for c in missing_cols:

        df[c] = 0
# We save the list of columns before we create new dummy columns

# to get the list of columns that have been created

old_col_train = list(train.drop(cat_feat, axis=1).columns)

old_col_test = list(test.drop(cat_feat, axis=1).columns)



# We create dummy variables for the train and test sets

train = dummify(train, cat_feat, True)

test = dummify(test, cat_feat, True)



# We list all the new columns. Those are the dummy columns

# that should appear in both dataframes

new_col_train = [c for c in list(train.columns) if c not in old_col_train]

new_col_test = [c for c in list(test.columns) if c not in old_col_test]



# Finally, we add the missing columns in both dataframes

add_missing_dummy_columns(test, new_col_train)

add_missing_dummy_columns(train, new_col_test)

# We control that both df have the same shape

print("Train: {}".format(train.shape))

print("Test: {}".format(test.shape))
model = GradientBoostingRegressor()
X = train.drop(["ID","y"], axis=1)

y = train["y"]
def plot_learning_curves(estimator, X, y, scoring="accuracy", cv=None, n_jobs=1, train_sizes=np.linspace(0.1,1.0,5)):

    """ Generate a plot showing training and test learning curves

        source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html



    Parameters

    ----------

    estimator: object type

        the estimator that will be used to implement "fit" and "predict"



    X: array, shape(n_samples, m_features)

        Training vector



    y: array, shape(n_samples)

        Target relative to X



     scoring:string

        The scoring method   



    cv: int

        Cross-validation splitting strategy



    n_jobs: int

        Number of jobs to run in parallel



    train_sizes: array, shape(n_ticks)

        Number of training examples that will be used to generate

        the learning curve

    """



    plt.figure()

    plt.title("Learning Curves\n")

    plt.xlabel("Training examples")

    plt.ylabel("Score ({})".format(scoring))

    plt.legend(loc="best")

    plt.grid()



    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)



    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")



    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1,

                     color="g")



    plt.plot(train_sizes, train_scores_mean, "o-", color="r",

             label="Training score")



    plt.plot(train_sizes, test_scores_mean, "o-", color="g", 

            label="Cross-validation score")



    plt.show()
plot_learning_curves(model, X, y, scoring="R2", cv=10, n_jobs=4)
model.fit(X, y)
y_pred = model.predict(test.drop("ID", axis=1))



submission = pd.DataFrame()

submission["ID"] = test["ID"].values

submission["y"] = y_pred

submission.to_csv("gbr-2017-06-17.csv", index=False)