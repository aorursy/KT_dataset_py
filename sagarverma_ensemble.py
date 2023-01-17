import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

%matplotlib inline

df = pd.read_csv("../input/train.csv")
df.reset_index(drop=True, inplace=True)

df_test = pd.read_csv("../input/test.csv")
df.reset_index(drop=True, inplace=True)

