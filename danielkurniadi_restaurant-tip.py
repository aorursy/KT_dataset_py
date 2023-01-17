import numpy as np
import pandas as pd
pd.options.display.max_columns = 12

# Matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

# Seaborn
import seaborn as sns
sns.set()

# Graphic in SVG format are better
%config InlineBackend.figure_format = 'svg'

# Increase default plot size
from pylab import rcParams
rcParams['figure.figsize'] = (5, 4)
# Disable warnings
import warnings
warnings.simplefilter('ignore')
# Load dataset
tips_df = pd.read_csv("../input/tips.csv")

tips_df.head(8)
#Plot bill and tip distribution
features = ['total_bill', 'tip']
tips_df[features].plot(kind='density', subplots=True, layout=(1,2), sharey=False, figsize=(8,4));
#Plot bill and tip kde and gamma
_, axes = plt.subplots(1, 2, figsize=(8, 4))

for idx, feat in enumerate(features):
  sns.distplot(tips_df[feat], kde=True, hist=True, ax=axes[idx])
features = ['sex', 'smoker', 'day', 'time']

#Plot categorical features
_, axes = plt.subplots(2, 2, figsize=(9, 10), sharey=True)

for idx, feat in enumerate(features):
  ax = axes[idx//2][idx%2]
  g = sns.countplot(x=feat, data=tips_df, ax=ax);
  g.set_ylabel('customers count');
palettes = [["#DB4437", "#4285F4"],
           ["#4285F4", "#0F9D58"],
           ["#F4B400", "#DB4437"],
           ["#4285F4", "#F4B400"]]
# define function to plot total bill and tips over 4 days
def draw_barplot(x='day', y=('tip', 'total_bill'), data=None, hue=None, axes=[], palette=None):
  y1, y2 = y
  #tips average over 4 days and hue
  g1 = sns.barplot(x=x, y=y1, 
                 hue=hue, data=data, 
                 palette=palette, ax=axes[0])
  g1.set_ylabel('average tip');
  #total_bill average over 4 days and hue
  g2 = sns.barplot(x=x, y=y2,
                 hue=hue, data=data, 
                 palette=palette, ax=axes[1])
  g2.set_ylabel('average bill')
#create axes
_, axes = plt.subplots(1, 2, figsize=(10, 4))

draw_barplot(data=tips_df, axes=axes, palette=palettes[3])
hues = ['sex', 'time', 'smoker']
#create axes
_, axeses = plt.subplots(3, 2, figsize=(10, 15))
#plot
for idx,axes in enumerate(axeses):
  draw_barplot(data=tips_df, hue=hues[idx], axes=axes, palette=palettes[idx])

x, y = ('total_bill', 'tip')

#Plot
g = sns.jointplot(x=x, y=y, data=tips_df,
                 kind='scatter')
x, y = ('total_bill', 'tip')

# Pair Plot
sns.pairplot(tips_df, hue='sex');
# Pair Plot
sns.pairplot(tips_df, hue='time', palette=palettes[3]);
# Pair Plot
sns.pairplot(tips_df, hue='time', palette=palettes[2]);

