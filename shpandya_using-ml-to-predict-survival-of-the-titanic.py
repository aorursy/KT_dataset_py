import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
trainDF = pd.read_csv('../input/train.csv')
trainDF.size
trainDF.head()
maleDF = trainDF[trainDF.Sex == 'male']

femaleDF = trainDF[trainDF.Sex == 'female']