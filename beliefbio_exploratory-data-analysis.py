df.shape
import pandas as pd

df = pd.read_csv("../input/mental-heath-in-tech-2016_20161114.csv")

df.columns.tolist()
df.columns = ["V" + str(x) for x in range(63)]

df.head()
df.info()