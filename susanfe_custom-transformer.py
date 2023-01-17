import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_threshold_pct=None):
        self._cat_thld_pct = cat_threshold_pct
        
    def fit(self, X, y=None):
        self._column_names = X.columns.values
        # find categorial and numerical columns
        self._column_types = X.dtypes
        self._cat_columns = self._column_names[self._column_types=='object']
        self._num_columns = self._column_names[self._column_types!='object']
        self._feature_names = self._num_columns
        
        # categorial dictionary
        self._cat_dict={}
        for cat_col in self._cat_columns:
            cat_values = X[cat_col].value_counts()/yield_data[cat_col].count()
            if self._cat_thld_pct is not None:
                cat_values = cat_values[cat_values>self._cat_thld_pct]
            self._cat_dict[cat_col] = cat_values.index
            self._feature_names = np.append(self._feature_names, cat_col+'_'+cat_values.index)
            
        # numerical stats
        self._num_fill = X[self._num_columns].agg('median')
        self._num_means = X[self._num_columns].agg('mean')
        self._num_std = X[self._num_columns].agg('std')
        return self
        
    def transform(self, X):
        # standard transformation for numerical features
        data = (X[self._num_columns] - self._num_means)/self._num_std
        # turn categorial features into dummy variables
        for col in self._cat_columns:
            for cv in self._cat_dict[col]:
                data[col+'_'+cv] = X[col].apply(lambda x: int(x==cv))
        
        return data
    
    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)