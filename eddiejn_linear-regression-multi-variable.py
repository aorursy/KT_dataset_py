import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('../input/hiring.csv')
df
rego = linear_model.LinearRegression()
rego.fit(df.drop("salary",axis='columns'), df.salary)
rego.predict([[0,7,5]])
rego.coef_
