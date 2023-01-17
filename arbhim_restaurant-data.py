# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('../input/restaurant-scores-lives-standard.csv')

data.head()
data.info()
data.describe()
sns.boxplot(x='inspection_score', data = data, palette='viridis')
