import numpy as np # linear algebra

from numpy import arange

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sb
ifs = pd.read_csv("../input/international_financial_statistics_data.csv")
ifs.describe()