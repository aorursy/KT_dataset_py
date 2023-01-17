import pandas as pd

import numpy as np



df = pd.DataFrame({'c1':[np.random.rand() for i in range(10)],'c2':[np.random.rand() for i in range(10)]})

df.to_csv('test.csv')