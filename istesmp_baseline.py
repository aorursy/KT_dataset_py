import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale

from sklearn.linear_model import LogisticRegression
#IMPORT TRAIN DATA

df_train=pd.read_csv("/kaggle/input/isteml2020/train.csv")

df_train.head()
#IMPORT TEST DATA

df_test=pd.read_csv("/kaggle/input/isteml2020/test.csv")

df_test.head()
out=pd.DataFrame()

out["Id"]=range(0,4000)

out["Predicted"]=pd.DataFrame(pred[:,])

out.head()
out.to_csv('results_out.csv', index=False)