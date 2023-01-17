import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import StratifiedKFold

from sklearn import preprocessing

    

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/HR_comma_sep.csv')
df.shape
df.head()
df_long = pd.melt(df, value_vars=list(df.columns[0:8]))

g = sns.FacetGrid(df_long, col="variable", col_wrap = 2, 

                  sharey=False, sharex=False)

g.map(plt.hist, "value");
df.salary.value_counts()
df.sales.value_counts()
sns.factorplot(x='sales', y='left', hue='salary', ci=68, data=df, aspect=2)
sns.heatmap(df.corr(), annot=True, fmt=".2f")
X = df.drop(['left', 'salary', 'sales'], axis=1)

X['salary_q'] = df.salary.replace(to_replace={"low": 1, "medium": 2, "high": 3})

X = pd.concat([X, pd.get_dummies(df.sales)], axis=1)



y = df.left
X = pd.concat([X, pd.get_dummies(df.sales)], axis=1)
X.head()
X.dtypes
X = df.drop(['left'], axis=1)

y = df.left
skf = StratifiedKFold(n_splits=10)

for train, test in skf.split(X, y):

    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]



    # preprocess the data

    scaler = preprocessing.StandardScaler().fit(X_train)

    

    scaler.transform(X_train)

    
enc = preprocessing.OneHotEncoder()