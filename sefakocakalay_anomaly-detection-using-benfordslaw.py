import numpy as np
import pandas as pd
import matplotlib as plt
!pip install benfordslaw
df = pd.read_csv("../input/2016-us-election/primary_results.csv")
df.head()
df.info()
df.describe().T
df.isnull().any()
from benfordslaw import benfordslaw
bl = benfordslaw()
X = df['votes'].loc[df['candidate']=='Bernie Sanders'].values
print(X)
results = bl.fit(X)
bl.plot(title='Bernie Sanders')
