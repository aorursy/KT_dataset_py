import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
df_train = pd.read_csv('../input/train.csv', nrows=5)
df_test = pd.read_csv('../input/test.csv', nrows=5)
print(df_train)
print(df_test)
df_cliente =pd.read_csv('../input/cliente_tabla.csv')
print(df_cliente)
