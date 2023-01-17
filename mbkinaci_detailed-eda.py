import numpy as np
import pandas as pd 
import scipy as sci
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
print(os.listdir("../input"))
#Changing the working directory
os.chdir("../input/train_1")
len(os.listdir())
os.listdir()[:5]
types_of_csvs = [x.split('-')[1].split('.')[0] for x in os.listdir()]
set(types_of_csvs)
cells = pd.read_csv('event000001000-cells.csv')
hits = pd.read_csv('event000001000-hits.csv')
particles = pd.read_csv('event000001000-particles.csv')
truth = pd.read_csv('event000001000-truth.csv')
cells.shape
hits.shape
particles.shape
truth.shape
import missingno as msno
msno.bar(cells,figsize=(6,3))
cells.head()
cells.describe()
cells['hit_id'].value_counts().sort_values(ascending=False)[:10].plot(kind='bar',figsize=(10,5))
ax = sns.jointplot(x="ch0", y="ch1", data=cells.sample(5000), size=10)
sns.distplot(np.log(cells['value'] + 0.0001),kde=True,color='r',bins=100).set_title('Log Transformed Histogram of Value Column')
corr = cells.corr()

# Set up the matplot figure
f,ax = plt.subplots(figsize=(8,6))

#Draw the heatmap using seaborn
sns.heatmap(corr, cmap='inferno', annot=True)
hits.head()
hits.describe()
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10
colors=['purple', 'c', 'y', 'm', 'r','pink','orange']
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(111, projection='3d')

ax.plot(hits[hits['layer_id'] == 2]['x'], hits[hits['layer_id'] == 2]['y'], hits[hits['layer_id'] == 2]['z'], 'x', color=colors[0], label='2')
ax.plot(hits[hits['layer_id'] == 4]['x'], hits[hits['layer_id'] == 4]['y'], hits[hits['layer_id'] == 4]['z'], 'o', color=colors[1], label='4')
ax.plot(hits[hits['layer_id'] == 6]['x'], hits[hits['layer_id'] == 6]['y'], hits[hits['layer_id'] == 6]['z'], '.', color=colors[2], label='6')
ax.plot(hits[hits['layer_id'] == 8]['x'], hits[hits['layer_id'] == 8]['y'], hits[hits['layer_id'] == 8]['z'], '^', color=colors[3], label='8')
ax.plot(hits[hits['layer_id'] == 10]['x'], hits[hits['layer_id'] == 10]['y'], hits[hits['layer_id'] == 10]['z'], '+', color=colors[4], label='10')
ax.plot(hits[hits['layer_id'] == 12]['x'], hits[hits['layer_id'] == 12]['y'], hits[hits['layer_id'] == 12]['z'], 'v', color=colors[5], label='12')
ax.plot(hits[hits['layer_id'] == 14]['x'], hits[hits['layer_id'] == 14]['y'], hits[hits['layer_id'] == 14]['z'], '_', color=colors[6], label='14')

plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=18, bbox_to_anchor=(0, 0))

plt.show()
rcParams['figure.figsize'] = 10,5
hits['volume_id'].value_counts().plot(kind='bar',title = 'Volume ID bar Plot')
hits['layer_id'].value_counts().plot(kind='bar',title = 'Layer ID bar Plot')
hits['module_id'].value_counts()[:20].plot(kind='bar',title = 'Module ID bar Plot')
particles.head()
for i in list(particles.columns.values):
    print(i, len(set(particles[i])))
rcParams['figure.figsize'] = 20, 10
colors=['purple', 'orange']
ax = plt.subplot(111, projection='3d')

ax.plot(particles[particles['q'] == -1]['px'], particles[particles['q'] == -1]['py'], particles[particles['q'] == -1]['pz'], '.', color=colors[0], label='-1')
ax.plot(particles[particles['q'] == 1]['px'], particles[particles['q'] == 1]['py'], particles[particles['q'] == 1]['pz'], '_', color=colors[1], label='1')
plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=18, bbox_to_anchor=(0, 0))

plt.show()
pd.crosstab(particles['q'], particles['nhits'])
truth.head()
from pylab import rcParams
rcParams['figure.figsize'] = 30, 15
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(111, projection='3d')

ax.plot(truth['tpx'], truth['tpy'], truth['tpz'], '.')
plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=18, bbox_to_anchor=(0, 0))

plt.show()
np.log(truth['weight']+ 0.00001).plot(kind='hist',bins=50,title='Log Transformed Weight')