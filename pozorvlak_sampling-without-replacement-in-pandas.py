import numpy as np

import pandas as pd
data = np.random.randint(0, 1000, size=(1000000, 8))
df = pd.DataFrame(data=data, columns=list('abcdefgh'))
%%timeit

df.sample(100, replace=False)
%%timeit

df.sample(100, replace=True)
import random
def dumb_sample(df, n):

    chosen = set()

    max_i = df.shape[0] - 1

    size = 0

    while size < n:

        i = random.randint(0, max_i)

        if i not in chosen:

            chosen.add(i)

            size += 1

    return df.iloc[list(chosen)]
df2 = dumb_sample(df, 5)

df2
%%timeit

dumb_sample(df, 100)
%%timeit

df.sample(10000, replace=False)
%%timeit

df.sample(10000, replace=True)
%%timeit

dumb_sample(df, 10000)
%%timeit

df.sample(100000, replace=False)
%%timeit

df.sample(100000, replace=True)
%%timeit

dumb_sample(df, 100000)
df2 = pd.DataFrame(data=np.random.randint(0, 1000, size=(50000000, 8)), columns=list('abcdefgh'))
%%timeit

df2.sample(100, replace=False)
%%timeit

df2.sample(100, replace=True)
%%timeit

dumb_sample(df2, 100)
%%timeit

df2.sample(100000, replace=False)
%%timeit

df2.sample(100000, replace=True)
%%timeit

dumb_sample(df2, 100000)