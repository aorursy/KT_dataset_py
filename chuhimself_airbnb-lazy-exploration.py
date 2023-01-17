import numpy as np
import pandas as pd
import seaborn as sbs
import matplotlib.pyplot as plt
%matplotlib inline
df_train = pd.read_csv("../input/train_users_2.csv")
df_test = pd.read_csv('../input/test_users.csv')
sessions = pd.read_csv('../input/sessions.csv')

df_train.info()

