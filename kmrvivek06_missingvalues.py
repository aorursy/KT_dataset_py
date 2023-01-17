import pandas as pd

import os

import numpy as np
from sklearn.impute import SimpleImputer as Imputer

from sklearn.impute import KNNImputer

from sklearn.preprocessing import LabelEncoder
#List Files

os.listdir("../input/mis-data")
#Read Files

pymnt_df = pd.read_csv('../input/mis-data/payments_mis.csv')

cust_df = pd.read_csv('../input/customers/cust.csv')
pymnt_df.info()
X_df = pymnt_df['amount']
#Imputer

imputer = Imputer()

transformed_X = imputer.fit_transform(X_df.values.reshape(-1,1))
pymnt_df['Imputer'] = transformed_X


imputer = KNNImputer(n_neighbors=2)

df_filled = imputer.fit_transform(X_df.values.reshape(-1,1))

pymnt_df['KNNImputer'] = df_filled
pymnt_df.info()
pymt_df_cp = pymnt_df.copy()

pymnt_df['amount'] = pymnt_df['amount'].fillna(0)

pymnt_df[pymnt_df['amount'] == 0]
pymt_df_cp.info()
######Custom Imputer#########



class CustomImputer(BaseEstimator, TransformerMixin):

    def __init__(self, strategy='mean',filler='NA'):

        self.strategy = strategy

        self.fill = filler



    def fit(self, X, y=None):

        #if self.strategy in ['mean','median']:

            #if not all(X.dtypes == np.number):

                #raise ValueError('dtypes mismatch np.number dtype is \

                                 #required for '+ self.strategy)

        if self.strategy == 'mean':

            self.fill = X.mean()

        elif self.strategy == 'median':

            self.fill = X.median()

        elif self.strategy == 'mode':

            self.fill = X.mode().iloc[0]

        elif self.strategy == 'fill':

            if type(self.fill) is list and type(X) is pd.DataFrame:

                self.fill = dict([(cname, v) for cname,v in zip(X.columns, self.fill)])

        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)



custom_df = CustomImputer(strategy='mean').fit_transform(pymt_df_cp)

custom_df
custom_df.info()
cust_df.info()
from sklearn_pandas import CategoricalImputer

from sklearn.base import TransformerMixin

from sklearn.base import BaseEstimator
class DataFrameImputer(TransformerMixin):



    def __init__(self):

        """Impute missing values.



        Columns of dtype object are imputed with the most frequent value 

        in column.



        Columns of other types are imputed with mean of column.



        """

    def fit(self, X, y=None):



        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],

            index=X.columns)



        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)



xt = DataFrameImputer().fit_transform(cust_df)







xt
xt.info()