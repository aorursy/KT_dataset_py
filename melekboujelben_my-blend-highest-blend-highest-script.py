import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import minmax_scale

submission_1 = pd.read_csv("../input/blend_it_all.csv")
submission_2 = pd.read_csv("../input/submission_OOF.csv")

blend = submission_1.copy()
col = blend.columns

col = col.tolist()
col.remove('id')
blend[col] = 0.5*minmax_scale(submission_1[col].values)+0.5*minmax_scale(submission_2[col].values)

blend.to_csv("superblend.csv", index=False)