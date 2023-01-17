import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print("data source:", os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



%matplotlib inline
# Load dataset

bank = pd.read_csv("../input/bank.csv")

bank.head()
bank.info()
bank.describe()
# Explore our categorical values:



print(f"{'COLUMN':15} {'VALUES'} {'LEVELS':>60}")

print(f"{'------':15} {'------'} {'------':>60}")

for col in bank.select_dtypes(include='object').columns:

    print(f"{col:.<15} {str(bank[col].unique()):<60} {len(bank[col].unique()):.>6}")

    
# Remove the unneeded contact column:

bank.drop('contact', axis=1, inplace=True)

# Encode 'yes' and 'no' with 1 and 0

bank['deposit_cat'] = bank['deposit'].map({'yes': 1, 'no': 0})
# We can reduce the categories levels by unifying:

bank['poutcome'] = bank['poutcome'].replace(['other'], 'unknown')

bank['job'] = bank['job'].replace(['retired', 'student', 'unemployed', 'unknown'])
print(f"{'COLUMN':15} {'VALUES'} {'LEVELS':>60}")

print(f"{'------':15} {'------'} {'------':>60}")

for col in bank.select_dtypes(include='object').columns:

    print(f"{col:.<15} {str(bank[col].unique()):<60} {len(bank[col].unique()):.>6}")
bank.job.value_counts()
bank.marital.value_counts()
bank.education.value_counts()
bank.default.value_counts()
bank.housing.value_counts()
bank.loan.value_counts()
bank.month.value_counts()
bank.poutcome.value_counts()
bank.deposit.value_counts()
# Subscriptions per job category

jobs = list(bank['job'].unique())

for i in jobs:

    print(f"{i:.<15} {len(bank[(bank.deposit_cat == 1) & (bank.job == i)])}")
sns.catplot(x='job', kind='count', data=bank)
sns.catplot(x='marital', kind='count', data=bank)
sns.catplot(x='education', kind='count', data=bank)
sns.catplot(x='default', kind='count', data=bank)
sns.catplot(x='loan', kind='count', data=bank)
sns.catplot(x='housing', kind='count', data=bank)
bank.hist(bins=100, figsize=(20, 15))
corr_matrix = bank.corr()

corr_matrix
scatter_matrix(bank[['age', 'balance', 'duration', 'pdays']], figsize=(12, 8))
bank_enc = pd.get_dummies(data=bank, columns=['job', 'marital', 'education', 'poutcome', ], prefix=['job', 'marital', 'education', 'poutcome'])

bank_enc.columns



bank_enc['default_cat'] = bank_enc['default'].map({'yes': 1, 'no': 0})

bank_enc.drop('default', axis=1, inplace=True)

bank_enc['housing_cat'] = bank_enc['housing'].map({'yes': 1, 'no': 0})

bank_enc.drop('housing', axis=1, inplace=True)

bank_enc['loan_cat'] = bank_enc['loan'].map({'yes': 1, 'no': 0})

bank_enc.drop('loan', axis=1, inplace=True)
bank_enc[bank_enc['deposit_cat'] == 1].describe()
print("people subscribed to deposit and with loans:", len(bank_enc[(bank_enc.deposit_cat == 1) & (bank_enc.loan_cat) & (bank_enc.housing_cat)]))
print("People signed up to a term deposite with a credit default: ", len(bank_enc[(bank_enc.deposit_cat == 1) & (bank_enc.default_cat == 1)]))
plt.figure(figsize = (10,6))

sns.barplot(x='job', y='deposit_cat', data=bank)
corr = bank_enc.corr()
# Heatmap

plt.figure(figsize = (10,10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .82})

plt.title('Heatmap of Correlation Matrix')
# copy and encode to a new dataset from original bank.csv, to feed ML:

ml_data = pd.get_dummies(data=bank, columns=['job', 'marital', 'education', 'poutcome', ], prefix=['job', 'marital', 'education', 'poutcome'])



# separate predictors from target variables:

X = ml_data.drop(['deposit', 'deposit_cat'], axis=1)



X['default_cat'] = X['default'].map({'yes': 1, 'no': 0})

X.drop('default', axis=1, inplace=True)

X['housing_cat'] = X['housing'].map({'yes': 1, 'no': 0})

X.drop('housing', axis=1, inplace=True)

X['loan_cat'] = X['loan'].map({'yes': 1, 'no': 0})

X.drop('loan', axis=1, inplace=True)



# Drop 'month' and 'day' as they do not add value to our current investigation.

X.drop('month' , axis=1, inplace=True)

X.drop('day', axis=1, inplace=True)



Y = ml_data.deposit_cat.copy()

X.head()
Y.head()
from sklearn.model_selection import train_test_split

from sklearn import tree
# Split our ML dataset to 80% training data, keeping 20% for testing:

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Decision Tree model with depth=2

dt = tree.DecisionTreeClassifier(random_state=1, max_depth=2)

dt.fit(X_train, Y_train)

dt_score_train = dt.score(X_train, Y_train)

print("Training Score: ", dt_score_train)

dt_score_test = dt.score(X_test, Y_test)

print("Testing Score: ", dt_score_test)
some_data = X_test.iloc[:15]

some_data
some_labels = Y_test.iloc[:15]

some_labels
print("Predictions:", dt.predict(some_data))
print("Labels:", list(some_labels))
from sklearn.metrics import mean_squared_error
predictions = dt.predict(X_test)



lin_mse = mean_squared_error(Y_test, predictions)

lin_mse
lin_rmse = np.sqrt(lin_mse)

lin_rmse
from sklearn.model_selection import cross_val_score
scores = cross_val_score(dt, X_train, Y_train, scoring='neg_mean_squared_error', cv=10)

tree_rmse_scores = np.sqrt(-scores)
scores
tree_rmse_scores
def display_scores(scores):

    print("Scores: ", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())
display_scores(tree_rmse_scores)