import sub_process
import pandas as pd

import numpy as np



titanic = pd.read_csv("../input/train.csv")
titanic.info()
titanic.describe()
titanic.head().T
titanic.isnull().any()