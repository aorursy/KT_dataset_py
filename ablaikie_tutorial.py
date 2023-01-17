import pandas as pd
import numpy as np

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('../input/train.csv', header=0)
df.describe()
df = df.drop(['PassengerId','Name','Ticket'], axis=1)
df.head()
df[df["Age"]>18][["Age","Sex","Survived"]].groupby("Age").mean()
