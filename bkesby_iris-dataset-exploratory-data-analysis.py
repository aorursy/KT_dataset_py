import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import parallel_coordinates
from IPython.core.display import HTML

% matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

iris = pd.read_csv('../input/Iris.csv')
iris = iris.drop('Id', axis=1)
iris.info()
iris.describe().round(0)
# Group data by class ("Species")
grouped = iris.groupby('Species')

print("Class size:\n", grouped.size())
print("\nClass averages:\n")
display(HTML(grouped.mean().to_html()))
features = iris.values[:,:-1]

# Define style for boxplot components
medianprops = {'color': 'darkblue', 'linewidth': 3}
boxprops = {'color': 'darkblue', 'linestyle': '-', 'linewidth': 2}
whiskerprops = {'color': 'darkblue', 'linestyle': '-', 'linewidth': 2}
capprops = {'color': 'darkblue', 'linestyle': '-', 'linewidth': 2}

# Box and whisker plots
fig, ax = plt.subplots(figsize=(16,12))
ttl = 'Box-whisker plot of Features'

box = ax.boxplot(features,
           patch_artist=True,
           medianprops=medianprops,
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)

for box in box['boxes']:
    box.set_facecolor('lightblue')

# Customize title
ax.set_title(ttl, fontsize=26)
plt.subplots_adjust(top=0.9)
ax.title.set_position((0.5, 1.02))
# Customize axis
labels = [x[:-2] for x in iris.columns[:-1].tolist()]
ax.set_xticklabels(labels)
ax.set_ylabel('Cm', fontsize=15);
# Jointplot of univariate histograms and bivariate scatterplots
sepal_jplot = sns.jointplot(data=iris, x='SepalLengthCm', y='SepalWidthCm', 
                      kind='scatter', stat_func=None, size=10)

petal_jplot = sns.jointplot(data=iris, x='PetalLengthCm', y='PetalWidthCm', 
                      kind='scatter', stat_func=None, size=10);
# Bivariate relation between each pair of features via a pairplot
sns.pairplot(data=iris, hue='Species', size=3);
# Parallel plot to reaffirm patterns of each class (petal size seperating setosa class)
fig, ax = plt.subplots(figsize=(16,12))
ttl = 'Species parallel plot according to features'

parallel_coordinates(iris, 'Species', colormap='tab10');
