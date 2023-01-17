import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv("../input/PoliceKillingsUS.csv")

df.head()
df = pd.read_csv("../input/PoliceKillingsUS.csv", encoding="windows-1252")

df.head()