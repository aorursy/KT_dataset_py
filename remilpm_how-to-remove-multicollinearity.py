import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import Imputer

print(os.listdir("../input"))

Data1=pd.read_csv("../input/Titanic.csv")

Data1.head()



# Any results you write to the current directory are saved as output.
Data2=Data1.drop("Name",axis=1)

Data2=Data2.drop("Sex",axis=1)

Data2=Data2.drop("Ticket",axis=1)

Data2=Data2.drop("Cabin",axis=1)

Data2=Data2.drop("Embarked",axis=1)
#Applying multicollinearity to remove columns which are dependent on each other

# From looking at documentation, values between 5 and 10 are "okay".

# Above 10 is too high and so should be removed.

from statsmodels.stats.outliers_influence import variance_inflation_factor



class ReduceVIF(BaseEstimator, TransformerMixin):

    def __init__(self, thresh=10, impute=True, impute_strategy='median'):

        # From looking at documentation, values between 5 and 10 are "okay".

        # Above 10 is too high and so should be removed.

        self.thresh = thresh

        

        # The statsmodel function will fail with NaN values, as such we have to impute them.

        # By default we impute using the median value.

        # This imputation could be taken out and added as part of an sklearn Pipeline.

        if impute:

            self.imputer = Imputer(strategy=impute_strategy)



    def fit(self, X, y=None):

        print('ReduceVIF fit')

        if hasattr(self, 'imputer'):

            self.imputer.fit(X)

        return self



    def transform(self, X, y=None):

        print('ReduceVIF transform')

        columns = X.columns.tolist()

        if hasattr(self, 'imputer'):

            X = pd.DataFrame(self.imputer.transform(X), columns=columns)

        return ReduceVIF.calculate_vif(X, self.thresh)



    @staticmethod

    def calculate_vif(X, thresh=5.0):

        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified

        dropped=True

        while dropped:

            variables = X.columns

            dropped = False

            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

            

            max_vif = max(vif)

            if max_vif > thresh:

                maxloc = vif.index(max_vif)

                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')

                X = X.drop([X.columns.tolist()[maxloc]], axis=1)

                dropped=True

        return X
# Remove columns having higher VIF factor ot having high multicollinearity

Mult_Coll = ReduceVIF()

Data3 = Mult_Coll.fit_transform(Data2)

Data3.head()

# ReduceVIF fit and ReduceVIF transform are having high multicollinearity and dropped