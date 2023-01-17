!pip uninstall pyclustertend -y

!pip install pyclustertend
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pyclustertend import vat, ivat

from sklearn.preprocessing import normalize



df = pd.read_csv("../input/world-happiness-report-2019.csv")
df[df.isnull().any(axis=1)]
pd.qcut(df.Corruption, 4)
df.Corruption = df.Corruption.fillna(111)
dft = df.fillna(-1)
X = dft[dft.columns[~dft.columns.isin(["Country (region)"])]]
X = normalize(X)
vat(X)
ivat(X)