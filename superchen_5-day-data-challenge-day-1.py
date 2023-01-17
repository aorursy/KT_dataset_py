import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv("../input/scrubbed.csv")
data.describe().transpose()
print("Just a small beginning :)")