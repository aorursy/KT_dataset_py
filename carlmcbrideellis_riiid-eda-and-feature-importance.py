!pip install dabl

import dabl

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

import warnings

warnings.filterwarnings('ignore')

colorMap = sns.light_palette("blue", as_cmap=True)
!wc -l ../input/riiid-test-answer-prediction/train.csv
import pandas as pd

train_data = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', nrows=100000)
dabl.detect_types(train_data)
train_data.describe().style.background_gradient(cmap=colorMap)
for col in train_data:

    print(col,len(train_data[col].unique()))
train_data.isnull().sum()
pd.set_option('display.max_columns', None)

pivot_table = pd.pivot_table(train_data, index=['user_answer']).style.background_gradient(cmap=colorMap)

pivot_table
pivot_table = pd.pivot_table(train_data, index=['answered_correctly']).style.background_gradient(cmap=colorMap)

pivot_table
pivot_table = pd.pivot_table(train_data, index=['prior_question_had_explanation']).style.background_gradient(cmap=colorMap)

pivot_table
dabl.plot(train_data, target_col="answered_correctly")
plt.figure(figsize = (15,6))

# set 300 bins, one bin for each of the 300 time values

ax = sns.distplot(train_data['prior_question_elapsed_time'], kde=False, bins=300)

ax.set_xlim(0,75000)

ax.set_xlabel("Histogram of 'prior_question_elapsed_time'",fontsize=18)

values = np.array([rec.get_height() for rec in ax.patches])

norm = plt.Normalize(values.min(), values.max())

colors = plt.cm.jet(norm(values))

for rec, col in zip(ax.patches, colors):

    rec.set_color(col)

plt.show();
print("The mean prior_question_elapsed_time is: %.1f" % train_data['prior_question_elapsed_time'].mean())

print("The modal value is                     :",train_data['prior_question_elapsed_time'].mode().squeeze())
print("Skew of prior_question_elapsed_time is:      %.2f" %train_data['prior_question_elapsed_time'].skew() )

print("Kurtosis of prior_question_elapsed_time is: %.2f" %train_data['prior_question_elapsed_time'].kurtosis() )
plt.figure(figsize = (15,6))

ax = sns.distplot(train_data['task_container_id'], kde=False, bins=563)

ax.set_xlim(0,1000)

ax.set_xlabel("Histogram of 'task_container_id'",fontsize=18)

values = np.array([rec.get_height() for rec in ax.patches])

norm = plt.Normalize(values.min(), values.max())

colors = plt.cm.jet(norm(values))

for rec, col in zip(ax.patches, colors):

    rec.set_color(col)

plt.show();
plt.figure(figsize = (15,6))

ax = sns.distplot(train_data['content_id'], kde=False, bins=341) # why 341? Because it is a factor of 32736 ;-)

ax.set_xlim(0,14000)

ax.set_xlabel("Histogram of 'content_id'",fontsize=18)

values = np.array([rec.get_height() for rec in ax.patches])

norm = plt.Normalize(values.min(), values.max())

colors = plt.cm.jet(norm(values))

for rec, col in zip(ax.patches, colors):

    rec.set_color(col)

plt.show();
train_data.corr().style.background_gradient(cmap='Oranges')
how_good = train_data[train_data['answered_correctly'] != -1].groupby('user_id').mean()
plt.figure(figsize = (15,6))

ax = sns.distplot(how_good['answered_correctly'], color='darkcyan',bins=50)

ax.set_xlabel("Plot of the ratio of correct to incorrect answers by user",fontsize=18)

ax.set_xlim(0,1)

values = np.array([rec.get_height() for rec in ax.patches])

norm = plt.Normalize(values.min(), values.max())

colors = plt.cm.jet(norm(values))

for rec, col in zip(ax.patches, colors):

    rec.set_color(col)

plt.show();
print("The best score is: %.1f" % (how_good['answered_correctly'].max()*100), "%")

print("The mean score is:  %.1f" % (how_good['answered_correctly'].mean()*100), "%")
fig_1 = px.scatter(how_good, x=how_good['prior_question_elapsed_time'], y=how_good['answered_correctly'], 

                   trendline="ols", marginal_y="violin", marginal_x="box",

                   title=("Scatter plot of results with respect to the prior question elapsed time"))

fig_1.show()
X_train       = train_data.drop("answered_correctly",axis=1)

X_train       = X_train.fillna(X_train.mean())

y_train       = train_data["answered_correctly"]



from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_features='auto', 

                                    min_samples_leaf=10)

classifier.fit(X_train, y_train)
import eli5

from eli5.sklearn import PermutationImportance

perm_import = PermutationImportance(classifier, random_state=1).fit(X_train, y_train)

# visualize the results

eli5.show_weights(perm_import, top=None, feature_names = X_train.columns.tolist())