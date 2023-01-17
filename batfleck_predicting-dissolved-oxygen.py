import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
bottle = pd.read_csv("/kaggle/input/calcofi/bottle.csv")
bottle = bottle[["Depthm", "T_degC", "O2ml_L", "Salnty"]]
bottle.columns = ["Depth", "Temp", "O2 Level", "Salinity"]
bottle.info()
bottle.describe()
sns.heatmap(bottle.corr(), cmap = "coolwarm")
sns.pairplot(bottle)
sns.distplot(bottle["Depth"])

from sklearn.model_selection import train_test_split

bottle_train, bottle_test = train_test_split(bottle, test_size = 0.2, random_state = 42)


from sklearn.base import BaseEstimator, TransformerMixin

class NaRemover(BaseEstimator, TransformerMixin):
    def __init__(self): # no *args or **kargs
        pass
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X):
        return X.dropna()

from sklearn.pipeline import Pipeline

full_pipeline = Pipeline([
    ('remove_na', NaRemover())
])

bottle_prepared = full_pipeline.fit_transform(bottle_train)
bottle_labels = bottle_prepared["O2 Level"].copy()

bottle_prepared = bottle_prepared.drop("O2 Level", axis=1)
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit_transform(bottle_prepared)
pd.DataFrame(bottle_prepared).info()
pd.DataFrame(bottle_labels).info()
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
from sklearn.model_selection import cross_val_score

scores = cross_val_score(lin_reg, bottle_prepared, bottle_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(lin_rmse_scores)


lin_reg.fit(bottle_prepared, bottle_labels)
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
scores = cross_val_score(tree_reg, bottle_prepared, bottle_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
tree_reg.fit(bottle_prepared, bottle_labels)
bottle_test = full_pipeline.fit_transform(bottle_test)
bottle_test_labels = bottle_test["O2 Level"].copy()

bottle_test = bottle_test.drop("O2 Level", axis=1)
final_predictions = tree_reg.predict(bottle_test)
final_mse = mean_squared_error(final_predictions, bottle_test_labels)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - bottle_test_labels) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(), scale=stats.sem(squared_errors)))
