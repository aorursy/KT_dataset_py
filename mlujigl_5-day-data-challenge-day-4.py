import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline

cereal = pd.read_csv('../input/cereal.csv')
cereal.describe()
cereal.head(3)
manuf = cereal['mfr'].unique().tolist()

len(manuf)