import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df_imports = pd.read_csv('../input/autoimports85/imports-85.csv', na_values = '?')
df_imports.head()
df_imports.info()
df_imports.dtypes.eq('object').sum()
df_imports_string = df_imports.select_dtypes(include=["object"]).copy()

df_imports_string.head()
df_imports_string.info()
df_imports_string.isna().sum()
df_imports_string.dropna(inplace=True)

df_imports_string.info()
df_imports_string["num-of-doors"].value_counts()
maping = {"num-of-doors": {"four": 4, "two": 2}}
df_imports_string.replace(maping, inplace=True)

df_imports_string.head()
df_imports_string["body-style"] = df_imports_string["body-style"].astype("category")

df_imports_string.dtypes
df_imports_string.info()
df_imports_string["body-style-cat"] = df_imports_string["body-style"].cat.codes

df_imports_string.head()
df_imports_string["drive-wheels"].unique()
pd.get_dummies(df_imports_string, columns = ["drive-wheels"]).head()