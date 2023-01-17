import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
!pip install sweetviz
import sweetviz
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train.head()
my_report = sweetviz.analyze([train,"Train"],target_feat= "SalePrice")
my_report.show_html('Report.html')