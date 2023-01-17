# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings(action='ignore')
# !pip install -i https://test.pypi.org/simple/ dominance-analysis
!pip install -U dominance-analysis
from dominance_analysis import Dominance
from dominance_analysis import Dominance_Datasets
breast_cancer_data=Dominance_Datasets.get_breast_cancer()
dominance_classification=Dominance(data=breast_cancer_data,target='target',top_k=15,objective=0,pseudo_r2="mcfadden")
dominance_classification.incremental_rsquare()
dominance_classification.plot_incremental_rsquare()
dominance_classification.dominance_stats()
dominance_classification.dominance_level()




