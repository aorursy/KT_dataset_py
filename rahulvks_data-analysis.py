import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
records = pd.read_csv("../input/scirate_quant-ph_2016.csv",dtype={"id": str})
records.columns
print(len(set(records['authors'])))
#records.groupby('authors').count()



records.groupby(['title','scites','authors']).agg({'scites':sum}).head(10)
records.groupby('authors')['scites'].sum().head(10)