import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



eq = pd.read_csv('../input/database.csv')

print(eq.head())

print(eq.shape)