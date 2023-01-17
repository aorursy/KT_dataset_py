# Estimate Boston housing values using pipeline and random forest regressor
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
bh_train = pd.read_csv('../input/boston_train.csv', index_col=0)
bh_test = pd.read_csv('../input/boston_test.csv', index_col=0)
corr_matrix = bh_train.corr()
corr_matrix['medv'].sort_values(ascending=False)
bh_feature = bh_train.drop('medv', axis=1)
bh_label = bh_train['medv']
list(bh_feature)
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
bh_pipeline = Pipeline([
    ('selector', DataFrameSelector(list(bh_feature))),
    ('std_scaler', StandardScaler())
])

bh_prepared = bh_pipeline.fit_transform(bh_feature)
bh_prepared_test = bh_pipeline.fit_transform(bh_test)
bh_prepared
X_train, X_test, y_train, y_test = train_test_split(bh_prepared, bh_label, test_size=0.3, random_state=42)
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)
bh_prediction = forest_reg.predict(X_test)
bh_prediction
mse = mean_squared_error(y_test, bh_prediction)
np.sqrt(mse)
forest_reg.score(X_test, y_test)
result = forest_reg.predict(bh_prepared_test)
result

