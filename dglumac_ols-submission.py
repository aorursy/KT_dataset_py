from io import StringIO

import requests

import json

import pandas as pd
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
import statsmodels.formula.api as sm

from sklearn import linear_model

%matplotlib inline

import matplotlib

import numpy as np

import matplotlib.pyplot as plt
model = sm.ols(formula="SalePrice ~ YearBuilt + MSSubClass + LotArea + Street + Neighborhood + Condition2 + BldgType + HouseStyle + OverallQual + RoofMatl + ExterQual + GrLivArea", data=train_data).fit()
results = model.predict(test_data)
result_df=pd.DataFrame(results,index=test_data['Id'],columns=["SalePrice"])

result_df.to_csv("output.csv",header=True,index_label='Id')