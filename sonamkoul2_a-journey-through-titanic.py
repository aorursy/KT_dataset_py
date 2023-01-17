import numpy as np

import pandas as pd

train = pd.read_csv('../input/train.csv')

test = pd.read_csv("../input/test.csv")

train.describe()
df = pd.DataFrame(np.random.randn(5,6))

df.index = ["a",'b','c','d','e']

df.columns = ['A','B','C','D','E','F']

df

def na_method(alg):

    while(alg.isnull().any().any()):

        na_r = alg[alg.isnull().T.any().T.any()]

        

        

        