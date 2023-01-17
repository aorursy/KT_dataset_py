# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
metadata = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
metadata.head(2)
data_filter = metadata[metadata.abstract.str.contains('blood type',regex=True, na=False)].reset_index(drop=True)
len(data_filter)
data_filter.abstract[2]