%matplotlib inline 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')

df_train  = pd.read_csv("../input/train_users.csv")
df_test    = pd.read_csv("../input/test_users.csv")