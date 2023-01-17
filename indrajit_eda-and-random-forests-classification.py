import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



df = pd.read_csv('../input/HR_comma_sep.csv')

df.rename(columns={'sales': 'roles'}, inplace=True)

df.head()
sns.set(style="white", palette="muted", color_codes=True)

f, ax = plt.subplots(10, 3, figsize = (40,50)) 



for i1, k in enumerate(df.roles.unique()):

    d = df[(df['roles'] == k)]

    for i2, j in enumerate(['satisfaction_level', 'last_evaluation', 'average_montly_hours']):

        ax[i1, 1].set_title('{}'.format(k), fontsize = 30)

        sns.distplot(d[j], color="m", ax=ax[i1, i2])

        plt.setp(ax, yticks=[])

        plt.tight_layout()

f.subplots_adjust(hspace=0.5)
#sns.set(style="white")

d = df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years']]

corr = d.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
f = {'satisfaction_level': 'mean', 'number_project': 'mean', 'average_montly_hours': 'mean', 'time_spend_company': 'mean'}

df.groupby('salary').agg(f)
f = {'left': ['count', 'sum']}

d = df.groupby('salary').agg(f)

d = d.reset_index()



sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots()



# Plot the total crashes

sns.set_color_codes("pastel")

sns.barplot(x=d['left']['count']-d['left']['sum'], y=d['salary'], data= d,

            label="Retained", color="b")



# Plot the crashes where alcohol was involved

sns.set_color_codes("muted")

sns.barplot(x=d['left']['sum'], y=d['salary'], data=d,

            label="Left", color="b")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlabel="Nuber of Employees")

sns.despine(left=True, bottom=True)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler



X = df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years', 'salary']]

X['salary'] = np.where(X['salary'] == 'high', 5, np.where(X['salary'] == 'medium', 3, 0))

y = df['left']



scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



clf = RandomForestClassifier(max_depth = 3).fit(X_train_scaled, y_train)



print('Accuracy of RF classifier on training set: {:.2f}'

     .format(clf.score(X_train_scaled, y_train)))

print('Accuracy of RF classifier on test set: {:.2f}'

     .format(clf.score(X_test_scaled, y_test)))