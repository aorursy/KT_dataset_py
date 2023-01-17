import numpy as np

import pandas as pd

import seaborn as sns
# read the dataset

data = pd.read_csv('../input/7210_1.csv')
data.describe()
sns.distplot(data['prices.amountMax'], bins=100).set_title('Price Histogram')