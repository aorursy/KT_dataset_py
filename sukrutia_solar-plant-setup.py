
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

data = pd.read_excel("../input/Geographic_Data_Set.xls", sheet_name="input")
X=data.drop(['city'], axis=1)
X=X.drop(['country'], axis=1)
X.head()
from pygeocoder import geocoder

