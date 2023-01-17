import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/HR_comma_sep.csv')

data.head(10)
# convert label features to integers

data["salary_level"] = data["salary"].map({"low":1, "medium":2, "high":3})
sns.set(style="white")



# Compute the correlation matrix

corr = data.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(5, 4))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(10, 220, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5,

            square=True, xticklabels=True, yticklabels=True,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
sns.set(style="white")

f, ax = plt.subplots(figsize=(5, 4))

sns.barplot(x=data.satisfaction_level,y=data.left,orient="h", ax=ax)
sns.set(style="darkgrid")

g = sns.FacetGrid(data, row="sales", col="left", margin_titles=True)

bins = np.linspace(0, 1, 13)

g.map(plt.hist, "satisfaction_level", color="steelblue", bins=bins, lw=0)
sns.set(style="white", palette="muted", color_codes=True)



# Set up the matplotlib figure

f, axes = plt.subplots(3, 3, figsize=(9,7))

sns.despine(left=True)



#people that left

leavers = data.loc[data['left'] == 1]



# Plot a simple histogram with binsize determined automatically

sns.distplot(leavers['satisfaction_level'], kde=False, color="b", ax=axes[0,0])

sns.distplot(leavers['salary_level'], bins=3, kde=False, color="b", ax=axes[0, 1])

sns.distplot(leavers['average_montly_hours'], kde=False, color="b", ax=axes[0, 2])

sns.distplot(leavers['number_project'], kde=False, color="b", ax=axes[1,0])

sns.distplot(leavers['last_evaluation'], kde=False, color="b", ax=axes[1, 1])

sns.distplot(leavers['time_spend_company'], kde=False, bins=5, color="b", ax=axes[1, 2])

sns.distplot(leavers['promotion_last_5years'],bins=10, kde=False, color="b", ax=axes[2,0])

sns.distplot(leavers['Work_accident'], bins=10,kde=False, color="b", ax=axes[2, 1])





plt.tight_layout()
#all key employees

key_employees = data.loc[data['last_evaluation'] > 0.7].loc[data['time_spend_company'] >= 3]

key_employees.describe()
#lost key employees

lost_key_employees = key_employees.loc[data['left']==1]

lost_key_employees.describe()

"Number of key employees: ", len(key_employees)

"Number of lost key employees: ", len(lost_key_employees)

"Percentage of lost key employees: ", round((float(len(lost_key_employees))/float(len(key_employees))*100),2),"%"
#filter out people with a good last evaluation

leaving_performers = leavers.loc[leavers['last_evaluation'] > 0.7]
sns.set(style="white")



# Compute the correlation matrix

corr = leaving_performers.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(5, 4))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(10, 220, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5,

            square=True, xticklabels=True, yticklabels=True,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
#filter out people with a good last evaluation

satisfied_employees = data.loc[data['satisfaction_level'] > 0.7]
sns.set(style="white")



# Compute the correlation matrix

corr = satisfied_employees.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(5, 4))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(10, 220, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5,

            square=True, xticklabels=True, yticklabels=True,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)