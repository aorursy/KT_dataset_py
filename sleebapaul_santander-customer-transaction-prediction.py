import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

%config Completer.use_jedi = False
data_test = pd.read_csv("../input/santander-customer-transaction-prediction-dataset/test.csv")

data_train = pd.read_csv("../input/santander-customer-transaction-prediction-dataset/train.csv")
# Print all the columns which is not present in test data but present in training data



for col in data_train.columns:

    if col not in data_test.columns:

        print("`{}` is not present in test data".format(col))
data_train.head()
data_train.info()
data_test.info()
print(data_train.shape)

print(data_test.shape)
idx = data_train.target.value_counts().index

vals = data_train.target.value_counts().values

fig, ax = plt.subplots()

explode = (0.1, 0)

ax.pie(vals, labels=idx, explode=explode, autopct='%1.1f%%')

ax.axis('equal')

ax.set_title('Santanber target labels')

plt.show()


corr = data_train.corr().abs()

corr[corr == 1] = 0

s = corr.unstack().sort_values(ascending=False)

print(s.head())
data_train.columns
skewList = []

for colName in data_train.columns:

    if colName not in ['ID_code', 'target']:

        skewList.append([colName, abs(data_train[colName].skew())])



skewList.sort(key=lambda x: x[1], reverse=True)



skewdf = pd.DataFrame.from_records(skewList, columns=['colName', 'Skewness'])
print(skewdf.head(10))
fig, ax = plt.subplots( figsize=(4,4), )



sns.distplot(data_train["var_44"], ax=ax, color='r')

ax.set_title('Distribution of var_44', fontsize=14)

ax.set_xlim([min(data_train["var_44"]), max(data_train["var_44"])])

fig.show()
var = 'var_44'

tmp = pd.concat([data_train['target'], data_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='target', y=var, data=tmp)

fig.axis(ymin= min(data_train[var]), ymax=max(data_train[var]));

f.show()
#missing data

total = data_train.isnull().sum().sort_values(ascending=False)

percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head()
print("Number of unique values in ID_code: ", data_train.ID_code.nunique())
non_transaction_df = data_train.loc[data_train['target'] == 0]

non_transaction_df.shape
transaction_df = data_train.loc[data_train['target'] == 1]

transaction_df.shape[0]
len(transaction_df)
# Since our classes are highly skewed we should make them equivalent in order to have 

# a normal distribution of the classes.



# Lets shuffle the data before creating the subsamples



# frac =1 sampling will help us to shuffle the dataframe

data_train = data_train.sample(frac=1)



# amount of fraud classes 20098 rows.

transaction_df = data_train.loc[data_train['target'] == 1]

non_transaction_df = data_train.loc[data_train['target'] == 0][:len(transaction_df)]



print("Shape of transaction df: ", transaction_df.shape)

print("Shape of non transaction df: ", non_transaction_df.shape)



normal_distributed_df = pd.concat([transaction_df, non_transaction_df])

# Shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)



print("Balanced data set dimension: ", new_df.shape)
new_df.drop("ID_code", inplace=True, axis=1)

data_test.drop("ID_code", inplace=True, axis=1)
new_df.shape
data_test.shape
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import KFold, StratifiedKFold

import numpy as np



X = new_df.drop('target', axis=1)

y = new_df['target']



sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)



for train_index, test_index in sss.split(X, y):

    print("Train:", train_index, "Test:", test_index)

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    print('-' * 100)

    print("\nShape of original_Xtrain: ", original_Xtrain.shape)

    print("\nShape of original_Xtest: ", original_Xtest.shape)

    print('-' * 100)

    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]



# We already have X_train and y_train for undersample data thats why I am using original to distinguish 

# and to not overwrite these variables.

# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)



# Check the Distribution of the labels





# Turn into an array

original_Xtrain = original_Xtrain.values

original_Xtest = original_Xtest.values

original_ytrain = original_ytrain.values

original_ytest = original_ytest.values



# See if both the train and test label distribution are similarly distributed

train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)

test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)



print('-' * 100)



print('Label Distributions: \n')

print(train_counts_label/ len(original_ytrain))

print(test_counts_label/ len(original_ytest))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import learning_curve





def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

   

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
log_reg = LogisticRegression()

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

title = "Santanber training results"

plot_learning_curve(log_reg, title, X.values, y.values, None, cv=cv,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))