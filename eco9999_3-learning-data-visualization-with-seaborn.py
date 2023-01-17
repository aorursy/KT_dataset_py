# Importing field
# This section same with 'Chapter 2' data adjusting section.
# With some shortcuts ;) , and brief extensions.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

# For not be disturbed.
warnings.filterwarnings('ignore')

# Reading...
data_set = pd.read_csv('../input/voice.csv')

# Converting... ( String to boolean)
data_set['label'] = [0 if each == 'male' else 1 for each in data_set['label']]

# Normalizing...
data_set = (data_set - np.min(data_set)) / (np.max(data_set) - np.min(data_set))

# Separeting...
label = data_set['label'].values

# Not Dropping for this chapter.
# data_set = data_set.drop(columns=['label'])
# It sets seaborn's parameters.
sns.set()

# This dataset has 21 features, I am going to reduce that for more understandable visual. 
# 5 features is enough.
new_data = data_set[['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR']]

# Creating 2 rows and 1 column for subplots.
f, axes = plt.subplots(2, 1)

# Converting dataframe to np array.
new_data_array = np.array(new_data)

# Drawing pyplot's violinplot on first axis. 
axes[0].violinplot(new_data_array)

# Setting title.
axes[0].set_title('Without seaborn')

# Drawing seaborn's violinplot on second axis. 
sns.violinplot(data=new_data, ax=axes[1])


# Setting title.
axes[1].set_title('With seaborn')

# For avoid subplots overlap.
plt.subplots_adjust(hspace=0.5)

plt.show()
# Creating index dataframe. It contains index of elements.

# Column's name
column = ['index']

# Column's values
values = list(range(data_set.shape[0]))

# Creating dataframe.
index_df = pd.DataFrame(columns=column, data=values)

# First values for check.
index_df.head()
# Creating zeros dataframe. It contains zeros.

# Column's name
column = ['zeros']

# Column's values
zeros = [0 for each in range(data_set.shape[0])]

# Creating dataframe.
zeros_df = pd.DataFrame(columns=column, data=zeros)

# First values for check.
zeros_df.head()
# Let's merge this dataframes to main dataset.
# Merging zeros dataframe to main dataset.
data_set = data_set.merge(zeros_df, right_index=True, left_index=True, how='right')

# Merging index dataframe to main dataset.
data_set = data_set.merge(index_df, right_index=True, left_index=True, how='right')

# As you can see my label feature is sorted, Because of that I am going to mix my dataset for tesing plots.

# Let's drop this index feature for a second, for preserve it from mixing.
data_set.drop(columns=['index'], inplace=True)

# I going to sort this dataset by 'meanfreq' feature, and it's going to mix the other features.
data_set = data_set.sort_values(by=['meanfreq'], ascending=True)

# Reseting dataframes own index.
data_set = data_set.reset_index(drop=True)

# Merging with 'index' feature again.
data_set = data_set.merge(index_df, right_index=True, left_index=True, how='right')

# First values for check.
print(data_set.columns.values)
# Almost all plotting functions has same keyword arguments.
# Most useful keyword arguments: (In my opinion)
# x is x axis values it takes feature name.
# y is y axis values it takes feature name.
# aplha is same with matplotlib. It is for transparency.
# hue is a keyword for, which feature is going to determine colors.
# sytle is a keyword for, which feature is going to determine point shapes.
# linewidth is a keyword points edge linewidth.
# col is a keyword for, which feature is going to determine to sepetare columns.
# kind is plotting kind 'scatter' and 'line' default is 'scatter'
# edgecolor is a keyword for points edges colors.
# palette is a keyword for color palettes. 'Reds','Greens','Set3','husl' also available.

# Few variations in the below.
# First variation, has diffrent shapes, diffrent colors.
sns.relplot(x='index', y='meanfun', hue='label', data=data_set, linewidth=0, 
            palette='Set2',style='label')
plt.show()

# It is the same thing with top lines.
# kind keyword arguments default value is 'scatter' that means if you don't use kind keyword it will use scatterplot.

# sns.scatterplot(x='index', y='meanfun', hue='label', data=data_set, linewidth=0, 
#             palette='Set2',style='label')
# plt.show()


# Second variation, has diffrent shapes, diffrent colors, transparency.
sns.relplot(x='index', y='mindom', hue='label', data=data_set,
            alpha=0.5,palette='Set1',style='label')
plt.show()
# Third variation, has diffrent colors, white edge color, transparency.
sns.relplot(x='index', y='sd', hue='label', data=data_set, linewidth=1,
            alpha=0.5,palette='Reds', kind='line')
plt.show()

# It is the same thing with top lines.
# kind keyword arguments value is 'line' that means it will use lineplot.

# sns.lineplot(x='index', y='sd', hue='label', data=data_set, linewidth=1,
#            alpha=0.5,palette='Reds')
# plt.show()
# Fourth variation, diffrent colors, transparency.
sns.relplot(x='index', y='maxdom', hue='label', data=data_set,linewidth=0,
            alpha=0.75, palette='Blues')
plt.show()
# This is a diffrent usage of relplot.(Actually this plot has no meaning for this dataset, just for fun :))
# size is a keyword for, which feature is going to determine points size.
# sizes is a keyword for size limits.
# I had to close that legend part for avoid overlap.
# I have to use linewidth=0 argument because default value is 1, I don't want any edge line.
sns.relplot(x="meanfreq", y="sd", hue="skew", size="kurt", palette='Set3', sizes=(10, 100), data=data_set, linewidth=0, legend =False)

plt.show()
# It is the most useful plot for supervised learning I think.
# Keyword arguments are almost same.

# It's creating two categories for x coordinate.
sns.catplot(x='label', y='dfrange', data=data_set, hue='label', linewidth=1, palette='Set1')
plt.show()
# For violin plot.
sns.catplot(x='label', y='median', data=data_set, hue='label', kind='violin', palette='Set2')
plt.show()
# For bar plot.
sns.catplot(x='label', y='mindom', data=data_set, hue='label', kind='bar', palette='Set3')
plt.show()
# For box plot.
sns.catplot(x='label', y='maxdom', data=data_set, hue='label', kind='box', palette='husl')
plt.show()
# stripplot.
# This is very useful for ploting distributions.

# jitter is a keyword argument for point frequence. it can be float or boolean.
# size is keyword argument for point size.
sns.stripplot(x='meanfreq', data=data_set, jitter=True, linewidth=0, palette='husl', size=2)
plt.show()
# jitter=0.45 is fitts this distribution to figure.
# label feature has two value. Because of that there is two values in x coordinate.
sns.stripplot(x='label', y='meanfun', hue='label', data=data_set, jitter=0.45, alpha=0.75, palette='husl', size=2)
plt.show()
# zeros feature has one value. Because of that there is one values in x coordinate.
# dodge keyword arguments value is False, it means the points is not separated from each other.
sns.stripplot(x='zeros', y='meanfun', hue='label', data=data_set, jitter=0.45, dodge=False, palette='husl', size=2)
plt.show()
# zeros feature has one value. Because of that there is one values in x coordinate.
# dodge keyword arguments value is True, it means the points is separated from each other.
sns.stripplot(x='zeros', y='meanfun', hue='label', data=data_set, jitter=0.45, dodge=True, palette='husl', size=2)
plt.show()
# boxplot.
# This is very useful for seeing distribution differences.

# Creating 1 rows and 2 column for subplots.
f, axes = plt.subplots(1, 2)

# Creating first column.
sns.boxplot(x="label", y="meanfun", data=data_set, ax=axes[0], palette='husl')

# Creating second column.
sns.boxplot(x="label", y="modindx", data=data_set, ax=axes[1], palette='husl')

# For avoid subplots overlap.
plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.show()
# kde_kws keyword argument is some kinda property dictionary.
# shade keyword argument is for drawing shade below distribution line.
# vertical keyword argument is for orientation of plot.
# hist keyword argument is for drawing histogram on it.

# Without histogram,has shade. orientation horizontal. 
sns.distplot(data_set['meanfun'], hist=False, kde_kws={"shade": True}, color='pink', label='meanfun', vertical=False)
plt.show()
# With histogram, no shade. orientation vertical. 
sns.distplot(data_set['sd'], hist=True, color='green', label='sd', vertical=True)
plt.show()
# (kernel density estimate)
# bw keyword argument is for drawing estimation accuracy.
sns.kdeplot(data_set['sd'], shade=True,bw=0.1)
sns.kdeplot(data_set['median'], shade=True,bw=0.1)
sns.kdeplot(data_set['meanfun'], shade=True,bw=0.1)
sns.kdeplot(data_set['mode'], shade=True,bw=0.1)

plt.show()

# Creating 2 rows and 1 column for subplots.
f, axes = plt.subplots(2, 1)

#Setting subplots title.
axes[0].set_title('meanfun-bw=0.05')

# Creating first plot. 
sns.kdeplot(data_set['meanfun'], shade=True, color='g', bw=0.05, ax=axes[0])

#Setting subplots title.
axes[1].set_title('median-bw=0.01')

# Creating second plot. 
sns.kdeplot(data_set['meanfun'], shade=True, color='g', bw=0.01, ax=axes[1])

# For avoid subplots overlap.
plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.show()