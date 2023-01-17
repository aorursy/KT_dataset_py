import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
happiness_report_2015 = pd.read_csv('../input/2015.csv')
happiness_report_2016 = pd.read_csv('../input/2016.csv')
happiness_report_2015.head()
# Plotting heatmap of pearson's correlation for 2015
fig, axes = plt.subplots(figsize=(10, 7))
corr = happiness_report_2015.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr,linewidths=1,annot=True, mask=mask, vmax=.3, square=True)
axes.set_title("2015")
sns.pairplot(happiness_report_2015[['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)']])
# Plotting heatmap of pearson's correlation for 2016
fig, axes = plt.subplots(figsize=(10, 7))
corr = happiness_report_2016.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr,linewidths=1,annot=True, mask=mask, vmax=.3, square=True)
axes.set_title("2016")
sns.pairplot(happiness_report_2016[['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)']])
#plt.plot(happiness_report_2015['Happiness Score'])
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(10, 7))
sns.distplot(happiness_report_2015['Happiness Score'],kde=True,ax=axes[0])
sns.distplot(happiness_report_2016['Happiness Score'],kde=True,ax=axes[1])
axes[0].set_title("Distribution of Happiness Score for 2015")
axes[1].set_title("Distribution of Happiness Score for 2016")
happiness_report_2015['Year'] = '2015'
happiness_report_2016['Year'] = '2016'
happiness_report_2015_2016 = pd.concat([happiness_report_2015[['Happiness Score','Region','Year']],happiness_report_2016[['Happiness Score','Region','Year']]])
happiness_report_2015_2016.head()
sns.set(font_scale=1.5)
fig, axes = plt.subplots(figsize=(20, 9))
sns.boxplot(y='Region',x='Happiness Score',hue='Year', data = happiness_report_2015_2016)

