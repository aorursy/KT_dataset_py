import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell

from multiprocessing import Pool

from itertools import product

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

InteractiveShell.ast_node_interactivity = "all"

%matplotlib inline
test = pd.read_csv('test.csv')

shop = pd.read_csv('shops.csv')

submission = pd.read_csv('sample_submission.csv')

sales = pd.read_csv('sales_train_v2.csv')

items = pd.read_csv('items.csv')

item_category = pd.read_csv('item_categories.csv')