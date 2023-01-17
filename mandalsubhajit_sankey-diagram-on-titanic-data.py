import pandas as pd
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df1 = df.groupby(['Pclass', 'Sex'])['Name'].count().reset_index()
df1.columns = ['source', 'target', 'value']
df1['source'] = df1.source.map({1: 'Pclass1', 2: 'Pclass2', 3: 'Pclass3'})
df2 = df.groupby(['Sex', 'Survived'])['Name'].count().reset_index()
df2.columns = ['source', 'target', 'value']
df2['target'] = df2.target.map({1: 'Survived', 0: 'Died'})
links = pd.concat([df1, df2], axis=0)
links
!pip install psankey
from psankey.sankey import sankey
import pandas as pd
import matplotlib
matplotlib.rcParams['figure.figsize'] = [50, 50]
import matplotlib.pyplot as plt


fig, ax = sankey(links, aspect_ratio=4/3, nodelabels=True, linklabels=True, labelsize=30, nodecmap='copper', nodealpha=0.5, nodeedgecolor='white')
plt.show()