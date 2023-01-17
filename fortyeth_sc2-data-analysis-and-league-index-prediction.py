# Import packages

import numpy as np

import pandas as pd

from pandas import DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split # Deprecated

from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree

print(check_output(["ls", "../input"]).decode("utf8"))
# Import dataset and see what type of data we're working with

df_raw = pd.read_csv('../input/starcraft.csv',sep=',')

df_raw.info()
# First 5 rows of data

df_raw.head()
df = df_raw.dropna()

df = df[df['LeagueIndex']!=8]



# Number of rows with incomplete data and / or with LeagueIndex value of 8

print("Number of incomplete data rows dropped: " + str(len(df_raw) - len(df)))

print("Remaining records = " + str(len(df)))
# Select Style

plt.style.use('fivethirtyeight')



# Create Figure

fig, ax = plt.subplots(1,2, figsize = (14,8))

fig.suptitle('Starcraft 2 League Player Distribution', fontweight='bold', fontsize = 22)



# Specify histogram attributes

bins = np.arange(0, 9, 1)

weights = np.ones_like(df['LeagueIndex']) / len(df['LeagueIndex'])



# Count Histogram

p1 = plt.subplot(1,2,1)

p1.hist(df['LeagueIndex'], bins=bins, align='left') # Pure Count

plt.xlabel('League Index', fontweight='bold')

plt.title('Count')



# Percentage Histogram

p2 = plt.subplot(1,2,2)

p2.hist(df['LeagueIndex'], bins=bins, weights = weights, align='left') # % of Total (weights)

plt.xlabel('League Index', fontweight='bold')

plt.title('Percentage', )

yvals = plt.subplot(1,2,2).get_yticks()

plt.subplot(1,2,2).set_yticklabels(['{:3.1f}%'.format(y*100) for y in yvals])



plt.show()
df.describe()
pd.unique(df['Age'])
# Looking at that 1 million TotalHours row

df[df['TotalHours']==1000000]
df_temp = df[['Age', 'TotalHours']].copy(deep=True)

df_temp['TotalHoursYears'] = df_temp['TotalHours'] / 24 / 365

df_temp['Age_Less_GP_Years'] = df_temp['Age'] - df_temp['TotalHoursYears']

df_temp.head()
df_temp[df_temp['Age_Less_GP_Years']<0]
# Removing that 1 million TotalHours row

df = df[df['TotalHours']!=1000000]

print('Remaining records in df= ' + str(len(df)))



# Deleting df_temp

del(df_temp)
# Create list of player attributes (potential features) for data analysis

lst_pf = list(df.columns)

lst_pf = lst_pf[2:20]

num_of_pf = len(lst_pf)

lst_counter = list(np.arange(1,num_of_pf + 1,1))

print(lst_pf)

print(lst_counter)
# Create and apply dictionary of League Index Names for future plot labels

dct_league_names = {1:'Bronze', 2:'Silver', 3:'Gold', 4:'Platinum', 5:'Diamond', 

                    6:'Master', 7:'Grandmaster'}

df['LeagueName'] = df['LeagueIndex'].map(dct_league_names)



# Check mapping was applied correctly

df[['LeagueIndex', 'LeagueName']].head(10)



pvt_num_lpt = df.pivot_table(index='LeagueName', values=['LeagueIndex'], aggfunc='count', 

                             margins=False)
# Setup graph formatting

plt.style.use('seaborn')

sns.set_palette("dark")

sns.set_style("whitegrid")



# Boxplots of the potential features

fig, axes = plt.subplots(num_of_pf, 1, sharex = True, figsize = (14,30))

fig.suptitle('Attribute Percentile Distributions', fontsize=22, 

             fontweight='bold')

fig.subplots_adjust(top=0.95)



for pf, c in zip(lst_pf, lst_counter):

    p = plt.subplot(num_of_pf,1,c) # (rows, columns, graph number)

    sns.boxplot(x = 'LeagueIndex', y = pf, data=df, showfliers=False) 

    # outliers excluded from plots given visual density, but data points have not been removed

    if c < num_of_pf: # remove xtick labels and xaxis title for all plots, excluding the last

        labels = [item.get_text() for item in p.get_xticklabels()]

        empty_string_labels = [''] * len(labels)

        p.set_xticklabels(empty_string_labels)

        p.set(xlabel='')

    if c== 1:

        p.set_title('Box and Whisker Plots\n', fontsize=16)

     

plt.show()
sns.set_palette("dark")

sns.set_style("whitegrid")



# Use Facetgrid to visualize distributions of UniqueUnitsMade across LeagueIndexes

g = sns.FacetGrid(df, col='LeagueIndex', col_wrap=3, margin_titles=True)

g.map(plt.hist, 'UniqueUnitsMade')

g.fig.suptitle('Unique Units Made', fontweight='bold', fontsize=16)

plt.subplots_adjust(top=0.90)
sns.set_palette("dark")

sns.set_style("whitegrid")



plt.hist(df['UniqueUnitsMade'][df['LeagueIndex']==7])

plt.title('Grandmaster League (Index = 7)')

plt.ylabel('Count')

plt.xlabel('Unique Units Made')

plt.show()
# Copy all df columns to new df, excluding GameID for easy matrix reading

df_for_r = df.copy(deep=True)

del df_for_r['GameID']



# Set figure style

plt.style.use('fivethirtyeight')



# Create figure

fig, axes = plt.subplots(nrows=1, ncols = 1, figsize = (14,10))

fig.suptitle('Attribute Relationships', fontsize=22, fontweight='bold')

# fig.subplots_adjust(top=0.95)



# Generate a mask to hide the upper triangle for a cleaner heatmap.  Less visual noise the better.

mask = np.zeros_like(df_for_r.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Create correlation matrix heatma[]

r_matrix = df_for_r.corr().round(decimals=1)

sns.heatmap(r_matrix, mask=mask, square=True, cmap='YlGnBu', linewidths=.5, annot=True, fmt='g', 

            annot_kws={'size':10})

axes.set_title('     Correlation Matrix\n')

plt.show()
lst_best_features = ['TotalHours', 'APM','SelectByHotkeys', 'AssignToHotkeys', 'NumberOfPACs',

                     'GapBetweenPACs', 'ActionLatency']
df.head()
# SelectKBest Work
x_features = df.copy(deep=True)

x_features = x_features[lst_best_features]

y_target = df['LeagueIndex']

x_features.head()
lin_model = svm.LinearSVC()

lin_model = lin_model.fit(x_features, y_target)

print('SVM Score:', lin_model.score(x_features, y_target))

predicted = lin_model.predict(x_features)

print(sorted(pd.unique(predicted)))
from sklearn.naive_bayes import GaussianNB

# Naive Bayes Model

bayes_model = GaussianNB()



bayes_model.fit(x_features, y_target)

bayes_model.score(x_features, y_target)
log_model = LogisticRegression()

log_model = log_model.fit(x_features, y_target)

log_model.score(x_features, y_target)

print('Log Score:', log_model.score(x_features, y_target)) 

predicted = log_model.predict(x_features) 

print(sorted(pd.unique(predicted)))
test_sizes = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size = test_sizes[0])

log_model = LogisticRegression()

log_model = log_model.fit(x_train, y_train)

log_model.score(x_train, y_train)

print('Log Score:', log_model.score(x_train, y_train)) 

predicted = log_model.predict(x_train) 

print(sorted(pd.unique(predicted)))
# KNN model

knn_model = KNeighborsClassifier(n_neighbors=15)

knn_model = knn_model.fit(x_features, y_target)

knn_model.score(x_features, y_target)

print('KNN Score:', knn_model.score(x_features, y_target))

predicted = knn_model.predict(x_features)

print(sorted(pd.unique(predicted)))
test_sizes = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size = test_sizes[5])

                                                    

# KNN Model by Training Size

knn_model = KNeighborsClassifier(n_neighbors=15)

knn_model = knn_model.fit(x_train, y_train)

knn_model.score(x_train, y_train)  
x_train.shape
# Decision tree model

tree_model = tree.DecisionTreeClassifier(splitter='best')

tree_model = tree_model.fit(x_features, y_target)

tree_model.score(x_features, y_target)

print('Tree Score:', tree_model.score(x_features, y_target))

predicted = tree_model.predict(x_features)

print(sorted(pd.unique(predicted)))
sorted(pd.unique(df['LeagueIndex']))