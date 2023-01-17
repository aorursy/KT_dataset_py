import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

import os
print(os.listdir("../input"))
data=pd.read_json("../input/recipes.json",lines=True)
data.head()
data.info()
data['Author'].value_counts()[0:10]
