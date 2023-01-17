import numpy as np

import pandas as pd

import gc



from memory_profiler import memory_usage
df = pd.DataFrame(np.random.rand(330000, 20))

'filesize: {} MB'.format(df.memory_usage().sum() // 1024**2)
def write():

    df.to_excel('rand.xlsx')
mem = memory_usage((write), interval=1)



ax = pd.DataFrame(mem, columns=['float64']).plot(grid=True)

ax.set_xlabel('elasped time[sec]')

ax.set_ylabel('memory_usage[MB]')
df = df.astype(np.float16)

'filesize: {} MB'.format(df.memory_usage().sum() // 1024**2)
mem = memory_usage((write), interval=1)



ax = pd.DataFrame(mem, columns=['float16']).plot(grid=True)

ax.set_xlabel('elasped time[sec]')

ax.set_ylabel('memory_usage[MB]')