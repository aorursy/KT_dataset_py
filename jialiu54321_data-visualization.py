import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import math

%matplotlib inline

train_users = pd.read_csv("../input/train_users_2.csv")
countries = pd.read_csv("../input/countries.csv")

train_users.replace('NDF', np.nan, inplace=True)
train_users.replace('other', np.nan, inplace=True)

countries.head(len(countries))