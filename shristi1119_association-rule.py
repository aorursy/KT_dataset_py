import numpy as np

import pandas as pd

from apyori import apriori

import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer
market=pd.read_csv("hfi_cc_2018.csv",na_values=["?",",","NaN"])
market.shape
market.head()
market.dtypes
market.isnull().sum()
market1=market.iloc[:, 1:4]
market1.head()
market1.ISO_code.value_counts()
market.countries.value_counts
market1.region.value_counts
#mode_imputer = Imputer(strategy='most_frequent')

#imputed_market1 = pd.DataFrame(mean_imputer.fit_transform(market1),columns=market1.columns)

market1=market1.fillna({"ISO_code":"ALB"})

market1=market1.fillna({"countries":"Zimbabwe"})
market1=market1.fillna({"region":"Sub-Saharan Africa"})
market1.isnull().sum()
#market1=market.drop('ISO_code',axis=1)

#market=pd.get_dummies(market)





market.drop(market.iloc[:, 1:4], inplace = True, axis = 1)
market.head()
mean_imputer = Imputer(strategy='mean')

imputed_market = pd.DataFrame(mean_imputer.fit_transform(market),columns=market.columns)

imputed_market.head()
imputed_market.isnull().sum()
type(market1)
type(imputed_market)
market_tot=pd.concat([market1,imputed_market],axis=1)

market_tot.head()
association_results=apriori(market_tot,min_support=0.0046,min_confidence=.2,min_lift=2,min_length=2)

association_results=list(association_results)
print(len(association_results))
print(association_results)