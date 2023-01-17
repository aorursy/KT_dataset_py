import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold

df=pd.read_csv("../input/train.csv")

df1=pd.read_csv("../input/test.csv")
rfc=RandomForestClassifier()
df.head()
rfc.fit(df[["Pclass"]],df["Survived"])
pre=rfc.predict(df1[["Pclass"]])
pre.shape

print(pre)