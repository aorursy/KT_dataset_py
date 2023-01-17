import pandas as pd

import numpy as np
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv",dtype={"Age": np.float64},)
Age1=pd.DataFrame({'age':np.arange(100),'score':np.zeros(100)})

Age1
b=np.zeros(100)

b