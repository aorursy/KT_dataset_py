import pandas as pd
sales = pd.read_csv('../input/sales_train.csv.gz', compression='gzip')
sales.item_cnt_day.mean()
import numpy as np

c_1,c_2 = 0.5,     1.242641
s_1,s_2 = 1.23646, 1.54960

X = np.array([[1., -2.*c_1],
              [1., -2.*c_2]])
S = np.array([s_1*s_1 - c_1*c_1,
              s_2*s_2 - c_2*c_2])

A, B = np.matmul(np.linalg.inv(X), S)
print("A=%.3f\tB=%.3f" % (A, B))
C  = B
print("C=%.3f"%C)
import math
std = math.sqrt(A - B*B)
print("std=%.3f"%std)