import numpy as np

import pandas as pd



d = {

    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15 ,16, 17, 18, 19, 20, 21, 22, 23, 24, 25],

    'sexo': ['M', 'M', 'M', 'F', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', np.nan, 'M', 'M', 'M', 'M', 'F', 'F', np.nan, 'F', 'F', np.nan, 'M', 'M'],

    'idade': [21, 19, np.nan, 21, 21, 23, 22, np.nan, 22, 23, 22, 22, 23, np.nan, 21, 23, 22, np.nan, 20, 22, 21, 23, np.nan, 22, 23],

    'salario': [2000.00, 1850.00, 2600.00, 2100.00, 2300.00, 2950.00, 2320.00, 3300.00, 2780.00, 3540.00, 2120.00, 2890.00, 2000.00, 3200.00, 2305.00, 2900.00, 2500.00, 3500.00, 1750.00, 3200.00, 2400.00, 4000.00, np.nan, 2390.00, 5100.00],

    'cargo_gerencia': [False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True],

    'departamento': ['Help Desk', 'Vendas', 'Help Desk', np.nan, 'Help Desk', 'Vendas', 'Vendas', 'Financeiro', 'Financeiro', 'Financeiro', 'RH', 'Financeiro', 'Financeiro', 'RH', np.nan, np.nan, 'Vendas', 'Vendas', 'RH', 'Help Desk', 'RH', 'RH', np.nan, 'Vendas', 'Vendas'],

    'tipo_sanguineo': [np.nan, 'A+', np.nan, np.nan, np.nan, 'B-', 'O+', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 'A+', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 'O-', np.nan]

}



df = pd.DataFrame.from_dict(d)

df.head()
len(df)
df.shape
df.describe()
df.describe(include='all')
df.info()
df.count()
df.count().where(lambda s: s < len(df)).dropna()
df.count().loc[lambda s: s < len(df)]
df.isna().sum()
df.isna().sum().where(lambda s: s > 0).dropna()
df.isna().sum().loc[lambda s: s > 0]
df.isna().sum() / len(df)
(df.isna().sum() / len(df)).where(lambda s: s > 0).dropna()
(df.isna().sum() / len(df)).loc[lambda s: s > 0]
df.isna().mean()
df.isna().mean().where(lambda s: s > 0).dropna()
df.isna().mean().loc[lambda s: s > 0]
df.isna().sum(axis=1)
df.isna().sum(axis=1).where(lambda s: s > 0).dropna()
df.isna().sum(axis=1).loc[lambda s: s > 0]
df.isna().sum(axis=1) / len(df.columns)
(df.isna().sum(axis=1) / len(df.columns)).where(lambda s: s > 0).dropna()
(df.isna().sum(axis=1) / len(df.columns)).loc[lambda s: s > 0]