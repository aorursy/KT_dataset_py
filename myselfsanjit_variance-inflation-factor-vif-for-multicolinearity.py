import numpy as np

import pandas as pd

import statsmodels.api as sm

import warnings

from pandas import DataFrame,Series

from scipy import stats

from sklearn.datasets import load_boston

warnings.filterwarnings('ignore')
boston = load_boston()

print (boston.DESCR)
X = boston["data"]

Y = boston["target"]

names = list(boston["feature_names"])
inp_df = pd.DataFrame(X, columns=names)

inp_df.head()
for i in range(0, len(names)):

    y = inp_df.loc[:, inp_df.columns == names[i]]

    x = inp_df.loc[:, inp_df.columns != names[i]]

    model = sm.OLS(y, x)

    results = model.fit()

    rsq = results.rsquared

    vif = round(1 / (1 - rsq), 2)

    print(f"R Square value of '{names[i]} column' is '{round(rsq, 2)}' keeping all other columns as features")

    print(f"Variance Inflation Factor of '{names[i]} column' is '{vif}' \n")