data = pd.read_csv('../input/breast-cancer-csv/breastCancer.csv')
from sklearn.datasets import load_breast_cancer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline
cells = load_breast_cancer()

print(cells.DESCR)
print(cells.target_names)
type(cells.data)

cells.data
cells.data.shape
import pandas as pd

raw_data=pd.read_csv('../input/breast-cancer-csv/breastCancer.csv')

raw_data.tail(10)