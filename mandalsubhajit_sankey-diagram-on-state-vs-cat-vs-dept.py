!pip install psankey
import pandas as pd

df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_evaluation.csv')
df1 = df.groupby(['state_id', 'cat_id'])['id'].count()
df2 = df.groupby(['cat_id', 'dept_id'])['id'].count()
df1, df2 = df1.reset_index(), df2.reset_index()
df1.columns = df2.columns = ['source', 'target', 'value']
links = pd.concat([df1, df2], axis=0)
links
from psankey.sankey import sankey
import matplotlib
matplotlib.rcParams['figure.figsize'] = [50, 50]
fig, ax = sankey(links, labelsize=30, nodecmap='copper')