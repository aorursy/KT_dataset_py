from numba import jit, cuda #for gpu

import numpy as np # linear algebra

import pandas as pd # data processing

import seaborn as sns

df=pd.read_csv("/kaggle/input/french-death/export.csv", sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
df.head(10)
try:df['Code du lieu de naissance']=pd.to_numeric(df['Code du lieu de naissance'], downcast="integer",errors="coerce")

except:pass

try:df['Code du lieu de deces']=pd.to_numeric(df['Code du lieu de deces'], downcast="integer",errors="coerce")

except:pass

try:df['SEXE']=pd.to_numeric(df['SEXE'], downcast="integer",errors="coerce")

except:pass

try:df["Numéro d'acte de deces"]=pd.to_numeric(df["Numéro d'acte de deces"], downcast="integer",errors="coerce")

except:pass

pd.options.display.float_format = '{:,.0f}'.format

try:df["Numéro d'acte de deces"]=df["Numéro d'acte de deces"].astype(int)

except:pass



df.info()
df.groupby(['Code du lieu de deces']).count()["NOM"].sort_values(ascending=False)
df.groupby(['Code du lieu de deces']).count()["NOM"].describe()
#histogram

sns.distplot(df.groupby(['Code du lieu de deces']).count()["NOM"]);
df.groupby(['PRENOM']).count()["NOM"].sort_values(ascending=False)