# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

!pip install -U dominance-analysis
# !pip install -U -i https://test.pypi.org/simple/ dominance-analysis==1.0.8
from dominance_analysis import Dominance_Datasets
from dominance_analysis import Dominance
boston_dataset=Dominance_Datasets.get_boston()
boston_dataset.head()
dominance_regression=Dominance(data=boston_dataset,target='House_Price',objective=1)
dominance_regression.incremental_rsquare()
dominance_regression.plot_incremental_rsquare()
dominance_regression.dominance_stats()
pd.set_option('display.max_colwidth', 200)
dominance_regression.dominance_level()


