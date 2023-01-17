import pandas as pd
pd.set_option('max_columns', None)
df = pd.read_csv("../input/recipeData.csv", encoding='latin-1').set_index("BeerID")
import missingno as msno
import matplotlib.pyplot as plt
msno.bar(df, figsize=(12, 6), fontsize=12, color='steelblue')
len(df), len(df.dropna())
# Here's a short recipe for a variable importance check:

# from sklearn.tree import DecisionTreeClassifier
# from yellowbrick.features import FeatureImportances

# clf = DecisionTreeClassifier()
# viz = FeatureImportances(clf)
# viz.fit(X_sample, y_sample)
# viz.poof()
df.shape[1], df.drop(['PrimingMethod', 'PrimingAmount'], axis='columns').shape[1]
df['MashThickness'].isnull().sum(), df['MashThickness'].fillna(df['MashThickness'].mean()).isnull().sum()
df['MashThickness'].mean(), df['MashThickness'].fillna(df['MashThickness'].mean()).mean()
# Format the data for applying ML to it.
popular_beer_styles = (pd.get_dummies(df['Style']).sum(axis='rows') > (len(df) / 100)).where(lambda v: v).dropna().index.values

dfc = (df
       .drop(['PrimingMethod', 'PrimingAmount', 'UserId', 'PitchRate', 'PrimaryTemp', 'StyleID', 'Name', 'URL'], axis='columns')
       .dropna(subset=['BoilGravity'])
       .pipe(lambda df: df.join(pd.get_dummies(df['BrewMethod'], prefix='BrewMethod')))
       .pipe(lambda df: df.join(pd.get_dummies(df['SugarScale'], prefix='SugarScale')))       
       .pipe(lambda df: df.assign(Style=df['Style'].map(lambda s: s if s in popular_beer_styles else 'Other')))
       .pipe(lambda df: df.join(pd.get_dummies(df['Style'], prefix='Style')))       
       .drop(['BrewMethod', 'SugarScale', 'Style'], axis='columns')
      )

c = [c for c in dfc.columns if c != 'MashThickness']
X = dfc[dfc['MashThickness'].notnull()].loc[:, c].values
y = dfc[dfc['MashThickness'].notnull()]['MashThickness'].values
yy = dfc[dfc['MashThickness'].isnull()]['MashThickness'].values
# Apply a regression approach to imputing the mash thickness.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import numpy as np
np.random.seed(42)
kf = KFold(n_splits=4)
scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    
    scores.append(r2_score(y_test, y_test_pred))

print(scores)
from fancyimpute import MICE

trans = MICE()
trans.complete
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create a sample point cloud.
X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.05, 0.10, 0.85],
                           class_sep=2, random_state=0)

# Select indices to drop labels from.
X_l = X.shape[0]
np.random.seed(42)
unl_idx = np.random.randint(0, len(X), size=X_l - 500)

# Back the labels up and drop them.
y = y.astype('float64')
X_train, y_train = X[unl_idx].copy(), y[unl_idx].copy()
X[unl_idx] = np.nan
y[unl_idx] = np.nan

# The fancyimpute package takes a single combined matrix as input. It differs in this from the X feature matrix, y response vector style of sklearn.
f = np.hstack((X, y[:, None]))

# Impute the missing values.
from fancyimpute import MICE
trans = MICE(verbose=False)
f_complete = trans.complete(f)
(f_complete == np.nan).any()