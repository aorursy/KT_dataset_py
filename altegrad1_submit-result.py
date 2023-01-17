

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import pandas as pd

df = pd.read_csv("/kaggle/input/random-forest/random_forest.csv")
df.head()
df.to_csv('random_forest.csv', index=False)