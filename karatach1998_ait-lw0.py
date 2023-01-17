import numpy as np

import pandas as pd
pa = np.array([0.95, 0.97, 0.99, 0.999, 0.9999])

p = 1/6
n = np.ceil(np.log(1 - pa) / np.log(1 - p))

n
experiment = np.random.rand(10000)

pe = np.sum(experiment <= p) / 10000

pe
ne = np.ceil(np.log(1 - pa) / np.log(1 - pe))

ne