import numpy as np # linear algebra

np.random.seed(666)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
