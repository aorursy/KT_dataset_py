!pip install mlfinlab
!pip install pyfolio
# Imports
import pandas as pd
import mlfinlab as ml
import sys
import mlfinlab.data_structures as ds
import numpy as np
import os
import datetime
import math
import sklearn as sk
from mlfinlab.datasets import (load_tick_sample, load_stock_prices, load_dollar_bar_sample)
import matplotlib.pyplot as plt
import pyfolio as pf
%matplotlib inline
pd.read_csv('kaggle/input/sample-data-from-mlfinlab-repo/stupid_data.csv')
