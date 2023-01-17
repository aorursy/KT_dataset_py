import pandas as pd

pd.options.mode.chained_assignment = None

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style = 'whitegrid')

import textwrap
data_score_humanities = pd.read_csv('../input/indonesia-college-entrance-examination-utbk-2019/score_humanities.csv')

data_score_science = pd.read_csv('../input/indonesia-college-entrance-examination-utbk-2019/score_science.csv')

data_universities = pd.read_csv('../input/indonesia-college-entrance-examination-utbk-2019/universities.csv')

data_majors = pd.read_csv('../input/indonesia-college-entrance-examination-utbk-2019/majors.csv')
score_humanities = data_score_humanities.copy()

score_humanities.head()
score_science = data_score_science.copy()

score_science.head()
data_majors.head()
data_universities.head()
print(score_humanities.info())

print('\n')

print(score_science.info())

print('\n')

print(data_majors.info())

print('\n')

print(data_universities.info())
#  First of all, calculate the average of total score

score_humanities['avg_score'] = score_humanities.iloc[:, 6:15].mean(axis = 1)

score_science['avg_score'] = score_science.iloc[:, 6:14].mean(axis = 1)
# Merge score_humanities with major and university dataframe to get major and university names

# First choice major and university

score_humanities = pd.merge(score_humanities, data_majors[['id_major','major_name']], left_on = 'id_first_major'

                            , right_on = 'id_major').drop(['id_major', 'id_first_major'], axis = 1)



score_humanities = pd.merge(score_humanities, data_universities[['id_university','university_name']], left_on = 'id_first_university'

                            , right_on = 'id_university').drop(['id_university', 'id_first_university'], axis = 1)



score_humanities['specific_first_choice'] = score_humanities['major_name'] +' - ' +score_humanities['university_name']

score_humanities['avg_score_first_choice'] = score_humanities.groupby('specific_first_choice')['avg_score'].transform('mean')



# Second choice major and university

# Initiate suffixes to give specific columns name because there are major and university name columns before, so it will be duplicated.

score_humanities = pd.merge(score_humanities, data_majors[['id_major','major_name']], left_on = 'id_second_major'

                            , right_on = 'id_major', suffixes=('_first_choice', '_second_choice')).drop(['id_major', 'id_second_major'], axis = 1)



score_humanities = pd.merge(score_humanities, data_universities[['id_university','university_name']], left_on = 'id_second_university'

                            , right_on = 'id_university', suffixes=('_first_choice', '_second_choice')).drop(['id_university', 'id_second_university'], axis = 1)



score_humanities['specific_second_choice'] = score_humanities['major_name_second_choice'] +' - ' +score_humanities['university_name_second_choice']

score_humanities['avg_score_second_choice'] = score_humanities.groupby('specific_second_choice')['avg_score'].transform('mean')

# Drop 'Unnamed: 0' columns, which is will not use in this project. Then, sort values by id_user

score_humanities = score_humanities.drop('Unnamed: 0', axis = 1).sort_values('id_user')

score_humanities.head()
# Merge score_science with major and university dataframe to get major and university names

# First choice major and university

score_science = pd.merge(score_science, data_majors[['id_major','major_name']], left_on = 'id_first_major', right_on = 'id_major'

                         , suffixes=('_first', '_second')).drop(['id_major', 'id_first_major'], axis = 1)



score_science = pd.merge(score_science, data_universities[['id_university','university_name']], left_on = 'id_first_university'

                            , right_on = 'id_university', suffixes=('_first_choice', '_second_choice')).drop(['id_university', 'id_first_university'], axis = 1)



score_science['specific_first_choice'] = score_science['major_name'] +' - ' +score_science['university_name']

score_science['avg_score_first_choice'] = score_science.groupby('specific_first_choice')['avg_score'].transform('mean')

# Second choice major and university

# Initiate suffixes to give specific columns name because there are major and university name columns before, so it will be duplicated.

score_science = pd.merge(score_science, data_majors[['id_major','major_name']]

         , left_on = 'id_second_major', right_on = 'id_major', suffixes=('_first_choice', '_second_choice')).drop(['id_major', 'id_second_major'], axis = 1)



score_science = pd.merge(score_science, data_universities[['id_university','university_name']], left_on = 'id_second_university'

                            , right_on = 'id_university', suffixes=('_first_choice', '_second_choice')).drop(['id_university', 'id_second_university'], axis = 1)



score_science['specific_second_choice'] = score_science['major_name_second_choice'] +' - ' +score_science['university_name_second_choice']

score_science['avg_score_second_choice'] = score_science.groupby('specific_second_choice')['avg_score'].transform('mean')

# Drop 'Unnamed: 0' columns, which is will not use in this project. Then, sort values by id_user

score_science = score_science.drop('Unnamed: 0', axis = 1).sort_values('id_user')

score_science.head()
# Create distribution plot on each columns

fig, ax = plt.subplots(5, 2, figsize = (14, 14))

fig.tight_layout(pad = 5)



# Define numeric columns on "Score Humanities" Data

num_score_humanities = ['score_eko', 'score_geo', 'score_kmb', 'score_kpu','score_kua'

                        , 'score_mat', 'score_ppu', 'score_sej', 'score_sos', 'avg_score']



for ax, n in zip(ax.flatten(), num_score_humanities):

    sns.distplot(ax = ax, a = score_humanities[n].dropna(), label = "Skewness : %.2f"%(score_humanities[n].skew()))

    ax.set_title(n, fontsize = 18)

    ax.legend(loc = 'best')



plt.show()
# Create heatmap data numeric

cormat = score_humanities[num_score_humanities].corr()

fig, ax = plt.subplots(figsize = (12, 8))

sns.heatmap(ax = ax, data = cormat, annot = True)

ax.set_yticklabels(cormat,rotation = 0)

plt.show()
# Define categorical columns on "Score Humanities" Data

first_choice_humanities = ['major_name_first_choice', 'university_name_first_choice']

first_choice_titles = ['Top 10 First Choices Humanities Majors', 'Top 10 First Choices University (Humanities Majors)']
fig, ax = plt.subplots(len(first_choice_humanities), 1, figsize = (14, 10))

fig.tight_layout(pad = 5)

max_width = 13

for ax, col, name in zip(ax.flatten(), first_choice_humanities, first_choice_titles):

    index = score_humanities[col].fillna('NaN').value_counts().index

    count = score_humanities[col].fillna('NaN').value_counts()

    sns.barplot(ax = ax, x = index, y = count, order = index[0:10])

    ax.set_title(name, fontsize = 18)

    ax.set_ylabel('Count', fontsize = 14)

    ax.set_xticklabels((textwrap.fill(x.get_text(), max_width) for x in ax.get_xticklabels()), fontsize = 10)

plt.show()
fig, ax = plt.subplots(len(first_choice_humanities), 1, figsize = (16, 12))

fig.tight_layout(pad = 10)

max_width = 13



for ax, col, name in zip(ax.flatten(), first_choice_humanities, first_choice_titles):

    values = score_humanities[col].value_counts().sort_values(ascending = False).index[0:10]

    top_data = score_humanities[score_humanities[col].isin(values)]

    sns.boxplot(ax = ax, data=top_data, x = top_data[col], y = top_data['avg_score'], order = values)

    ax.set_title(name, fontsize = 18)

    ax.set_xlabel('')

    ax.set_ylabel('AVG', fontsize = 14)

    ax.set_xticklabels((textwrap.fill(x.get_text(), max_width) for x in ax.get_xticklabels()), fontsize = 10)



plt.show()
fig, ax = plt.subplots(2, 1, figsize = (10, 8))

fig.tight_layout(pad = 5)



index = score_humanities['specific_first_choice'].fillna('NaN').value_counts().index

count = score_humanities['specific_first_choice'].fillna('NaN').value_counts()

sns.barplot(ax = ax[0], x = count, y = index, order = index[0:5])

ax[0].set_title('Top 5 First Choices Humanities Majors - University', fontsize = 16)

ax[0].set_xlabel('Count', fontsize = 14)

ax[0].set_ylabel('Humanities Majors - University', fontsize = 14)

ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize = 8)





values = score_humanities['specific_first_choice'].value_counts().sort_values(ascending = False).index[0:5]

top5_data = score_humanities[score_humanities['specific_first_choice'].isin(values)]

sns.boxplot(ax = ax[1], data=top5_data, x = top_data['avg_score'], y = top5_data['specific_first_choice'], order = values)

ax[1].set_title('Top 5 First Choices Humanities Majors - University', fontsize = 16)

ax[1].set_xlabel('Average Score', fontsize = 14)

ax[1].set_ylabel('Humanities Majors - University', fontsize = 14)

ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize = 8)

plt.show()
# Define categorical columns on "Score Humanities" Data

second_choice_humanities = ['major_name_second_choice', 'university_name_second_choice']

second_choice_titles = ['Top 10 Second Choices Humanities Majors', 'Top 10 Second Choices University (Humanities Majors)']
fig, ax = plt.subplots(len(first_choice_humanities), 1, figsize = (14, 10))

fig.tight_layout(pad = 5)

max_width = 13

for ax, col, name in zip(ax.flatten(), second_choice_humanities, second_choice_titles):

    index = score_humanities[col].fillna('NaN').value_counts().index

    count = score_humanities[col].fillna('NaN').value_counts()

    sns.barplot(ax = ax, x = index, y = count, order = index[0:10])

    ax.set_title(name, fontsize = 18)

    ax.set_ylabel('Count', fontsize = 14)

    ax.set_xticklabels((textwrap.fill(x.get_text(), max_width) for x in ax.get_xticklabels()), fontsize = 10)

plt.show()
fig, ax = plt.subplots(len(second_choice_humanities), 1, figsize = (16, 12))

fig.tight_layout(pad = 10)

max_width = 13



for ax, col, name in zip(ax.flatten(), second_choice_humanities, second_choice_titles):

    values = score_humanities[col].value_counts().sort_values(ascending = False).index[0:10]

    top_data = score_humanities[score_humanities[col].isin(values)]

    sns.boxplot(ax = ax, data=top_data, x = top_data[col], y = top_data['avg_score'], order = values)

    ax.set_title(name, fontsize = 18)

    ax.set_xlabel('')

    ax.set_ylabel('AVG', fontsize = 14)

    ax.set_xticklabels((textwrap.fill(x.get_text(), max_width) for x in ax.get_xticklabels()), fontsize = 10)



plt.show()
fig, ax = plt.subplots(2, 1, figsize = (10, 8))

fig.tight_layout(pad = 5)



index = score_humanities['specific_second_choice'].fillna('NaN').value_counts().index

count = score_humanities['specific_second_choice'].fillna('NaN').value_counts()

sns.barplot(ax = ax[0], x = count, y = index, order = index[0:5])

ax[0].set_title('Top 5 Second Choices Humanities Majors - University', fontsize = 16)

ax[0].set_xlabel('Count', fontsize = 14)

ax[0].set_ylabel('Humanities Majors - University', fontsize = 14)

ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize = 8)





values = score_humanities['specific_second_choice'].value_counts().sort_values(ascending = False).index[0:5]

top5_data = score_humanities[score_humanities['specific_second_choice'].isin(values)]

sns.boxplot(ax = ax[1], data=top_data, x = top5_data['avg_score'], y = top5_data['specific_second_choice'], order = values)

ax[1].set_title('Top 5 Second Choices Humanities Majors - University', fontsize = 16)

ax[1].set_xlabel('Average Score', fontsize = 14)

ax[1].set_ylabel('Humanities Majors - University', fontsize = 14)

ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize = 8)

plt.show()
# Create distribution plot on each columns

fig, ax = plt.subplots(5, 2, figsize = (14, 14))

fig.tight_layout(pad = 5)



# Define numeric columns on "Score Humanities" Data

num_score_science = ['score_bio', 'score_fis', 'score_kim', 'score_kmb','score_kpu'

                     , 'score_kua', 'score_mat', 'score_ppu', 'avg_score']



for ax, n in zip(ax.flatten(), num_score_science):

    sns.distplot(ax = ax, a = score_science[n].dropna(), label = "Skewness : %.2f"%(score_science[n].skew()))

    ax.set_title(n, fontsize = 18)

    ax.legend(loc = 'best')



plt.show()
# Create heatmap data numeric

cormat = score_science[num_score_science].corr()

fig, ax = plt.subplots(figsize = (12, 8))

sns.heatmap(ax = ax, data = cormat, annot = True)

ax.set_yticklabels(cormat,rotation = 0)

plt.show()
# Define categorical columns on "Score Humanities" Data

first_choice_science = ['major_name_first_choice', 'university_name_first_choice']

first_choice_titles = ['Top 10 First Choices Science Majors', 'Top 10 First Choices University (Science Majors)']
fig, ax = plt.subplots(len(first_choice_science), 1, figsize = (14, 10))

fig.tight_layout(pad = 5)

max_width = 13

for ax, col, name in zip(ax.flatten(), first_choice_science, first_choice_titles):

    index = score_science[col].fillna('NaN').value_counts().index

    count = score_science[col].fillna('NaN').value_counts()

    sns.barplot(ax = ax, x = index, y = count, order = index[0:10])

    ax.set_title(name, fontsize = 18)

    ax.set_ylabel('Count', fontsize = 14)

    ax.set_xticklabels((textwrap.fill(x.get_text(), max_width) for x in ax.get_xticklabels()), fontsize = 10)

plt.show()
fig, ax = plt.subplots(len(first_choice_science), 1, figsize = (16, 12))

fig.tight_layout(pad = 10)

max_width = 13



for ax, col, name in zip(ax.flatten(), first_choice_science, first_choice_titles):

    values = score_science[col].value_counts().sort_values(ascending = False).index[0:10]

    top_data = score_science[score_science[col].isin(values)]

    sns.boxplot(ax = ax, data=top_data, x = top_data[col], y = top_data['avg_score'], order = values)

    ax.set_title(name, fontsize = 18)

    ax.set_xlabel('')

    ax.set_ylabel('AVG', fontsize = 14)

    ax.set_xticklabels((textwrap.fill(x.get_text(), max_width) for x in ax.get_xticklabels()), fontsize = 10)



plt.show()
fig, ax = plt.subplots(2, 1, figsize = (10, 8))

fig.tight_layout(pad = 5)



index = score_science['specific_first_choice'].fillna('NaN').value_counts().index

count = score_science['specific_first_choice'].fillna('NaN').value_counts()

sns.barplot(ax = ax[0], x = count, y = index, order = index[0:5])

ax[0].set_title('Top 5 First Choices Science Majors - University', fontsize = 16)

ax[0].set_xlabel('Count', fontsize = 14)

ax[0].set_ylabel('Science Majors - University', fontsize = 14)

ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize = 8)





values = score_science['specific_first_choice'].value_counts().sort_values(ascending = False).index[0:5]

top5_data = score_science[score_science['specific_first_choice'].isin(values)]

sns.boxplot(ax = ax[1], data=top5_data, x = top_data['avg_score'], y = top5_data['specific_first_choice'], order = values)

ax[1].set_title('Top 5 First Choices Science Majors - University', fontsize = 16)

ax[1].set_xlabel('Average Score', fontsize = 14)

ax[1].set_ylabel('Science Majors - University', fontsize = 14)

ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize = 8)

plt.show()
# Define categorical columns on "Score Humanities" Data

second_choice_science = ['major_name_second_choice', 'university_name_second_choice']

second_choice_titles = ['Top 10 Second Choices Science Majors', 'Top 10 Second Choices University (Science Majors)']
fig, ax = plt.subplots(len(first_choice_science), 1, figsize = (14, 10))

fig.tight_layout(pad = 5)

max_width = 13

for ax, col, name in zip(ax.flatten(), second_choice_science, second_choice_titles):

    index = score_science[col].fillna('NaN').value_counts().index

    count = score_science[col].fillna('NaN').value_counts()

    sns.barplot(ax = ax, x = index, y = count, order = index[0:10])

    ax.set_title(name, fontsize = 18)

    ax.set_ylabel('Count', fontsize = 14)

    ax.set_xticklabels((textwrap.fill(x.get_text(), max_width) for x in ax.get_xticklabels()), fontsize = 10)

plt.show()
fig, ax = plt.subplots(len(second_choice_science), 1, figsize = (16, 12))

fig.tight_layout(pad = 10)

max_width = 13



for ax, col, name in zip(ax.flatten(), second_choice_science, second_choice_titles):

    values = score_science[col].value_counts().sort_values(ascending = False).index[0:10]

    top_data = score_science[score_science[col].isin(values)]

    sns.boxplot(ax = ax, data=top_data, x = top_data[col], y = top_data['avg_score'], order = values)

    ax.set_title(name, fontsize = 18)

    ax.set_xlabel('')

    ax.set_ylabel('AVG', fontsize = 14)

    ax.set_xticklabels((textwrap.fill(x.get_text(), max_width) for x in ax.get_xticklabels()), fontsize = 10)



plt.show()
fig, ax = plt.subplots(2, 1, figsize = (10, 8))

fig.tight_layout(pad = 5)



index = score_science['specific_second_choice'].fillna('NaN').value_counts().index

count = score_science['specific_second_choice'].fillna('NaN').value_counts()

sns.barplot(ax = ax[0], x = count, y = index, order = index[0:5])

ax[0].set_title('Top 5 Second Choices Science Majors - University', fontsize = 16)

ax[0].set_xlabel('Count', fontsize = 14)

ax[0].set_ylabel('Science Majors - University', fontsize = 14)

ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize = 8)





values = score_science['specific_second_choice'].value_counts().sort_values(ascending = False).index[0:5]

top5_data = score_science[score_science['specific_second_choice'].isin(values)]

sns.boxplot(ax = ax[1], data=top_data, x = top5_data['avg_score'], y = top5_data['specific_second_choice'], order = values)

ax[1].set_title('Top 5 Second Choices Science Majors - University', fontsize = 16)

ax[1].set_xlabel('Average Score', fontsize = 14)

ax[1].set_ylabel('Science Majors - University', fontsize = 14)

ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize = 8)

plt.show()
from sklearn.cluster import KMeans



data_cluster = score_humanities[['id_user', 'specific_first_choice', 'avg_score', 'avg_score_first_choice']]

X = data_cluster.iloc[:, 2:4].values

# I choose numbers of cluster based on the Elbow Method

kmeans = KMeans(n_clusters = 3, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)

y_kmeans = kmeans.fit(X)

y_kmeans = kmeans.predict(X)

data_cluster['clusters'] = y_kmeans

fig, ax = plt.subplots(figsize = (10, 8))

sns.scatterplot(ax = ax, data = score_science, x = data_cluster[data_cluster['clusters'] == 0]['avg_score_first_choice']

                , y = data_cluster[data_cluster['clusters'] == 0]['avg_score'], color = 'red', label = 'Likely to Be Rejected')

sns.scatterplot(ax = ax, data = score_science, x = data_cluster[data_cluster['clusters'] == 1]['avg_score_first_choice']

                , y = data_cluster[data_cluster['clusters'] == 1]['avg_score'], color = 'green', label = 'Likely to Be Accepted')

sns.scatterplot(ax = ax, data = score_science, x = data_cluster[data_cluster['clusters'] == 2]['avg_score_first_choice']

                , y = data_cluster[data_cluster['clusters'] == 2]['avg_score'], color = 'gray', label = 'Uncertainty')

sns.scatterplot(ax = ax, x = kmeans.cluster_centers_[:,1], y = kmeans.cluster_centers_[:,0], s= 200, color= 'black', label = 'Centroids')

ax.set_title('Clusters of Participants by Their Average Scores (Humanities Majors)', fontsize = 18)

ax.set_xlabel('Average Scores of Selected Majors and Universities')

ax.set_ylabel('Participants Average Scores')

plt.show()
data_cluster.loc[data_cluster['clusters'] == 0, 'clusters_information'] = 'Likely to Be Rejected'

data_cluster.loc[data_cluster['clusters'] == 1, 'clusters_information'] = 'Likely to Be Accepted'

data_cluster.loc[data_cluster['clusters'] == 2, 'clusters_information'] = 'Uncertainty'

cluster_first_choice_humanities = data_cluster.copy()

cluster_first_choice_humanities.head()
from sklearn.cluster import KMeans



data_cluster = score_humanities[['id_user', 'specific_second_choice', 'avg_score', 'avg_score_second_choice']]

X = data_cluster.iloc[:, 2:4].values

# I choose numbers of cluster based on the Elbow Method

kmeans = KMeans(n_clusters = 3, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)

y_kmeans = kmeans.fit(X)

y_kmeans = kmeans.predict(X)

data_cluster['clusters'] = y_kmeans

fig, ax = plt.subplots(figsize = (10, 8))

sns.scatterplot(ax = ax, data = score_science, x = data_cluster[data_cluster['clusters'] == 0]['avg_score_second_choice']

                , y = data_cluster[data_cluster['clusters'] == 0]['avg_score'], color = 'green', label = 'Likely to Be Accepted')

sns.scatterplot(ax = ax, data = score_science, x = data_cluster[data_cluster['clusters'] == 1]['avg_score_second_choice']

                , y = data_cluster[data_cluster['clusters'] == 1]['avg_score'], color = 'red', label = 'Likely to Be Rejected')

sns.scatterplot(ax = ax, data = score_science, x = data_cluster[data_cluster['clusters'] == 2]['avg_score_second_choice']

                , y = data_cluster[data_cluster['clusters'] == 2]['avg_score'], color = 'gray', label = 'Uncertainty')

sns.scatterplot(ax = ax, x = kmeans.cluster_centers_[:,1], y = kmeans.cluster_centers_[:,0], s= 200, color= 'black', label = 'Centroids')

ax.set_title('Clusters of Participants by Their Average Scores (Humanities Majors)', fontsize = 18)

ax.set_xlabel('Average Scores of Selected Majors and Universities')

ax.set_ylabel('Participants Average Scores')

plt.show()
data_cluster.loc[data_cluster['clusters'] == 0, 'clusters_information'] = 'Likely to Be Accepted'

data_cluster.loc[data_cluster['clusters'] == 1, 'clusters_information'] = 'Likely to Be Rejected'

data_cluster.loc[data_cluster['clusters'] == 2, 'clusters_information'] = 'Uncertainty'

cluster_second_choice_humanities = data_cluster.copy()

cluster_second_choice_humanities.head()
from sklearn.cluster import KMeans



data_cluster = score_science[['id_user', 'specific_first_choice', 'avg_score', 'avg_score_first_choice' ]]

X = data_cluster.iloc[:, 2:4].values

# I choose numbers of cluster based on the Elbow Method

kmeans = KMeans(n_clusters = 3, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)

y_kmeans = kmeans.fit(X)

y_kmeans = kmeans.predict(X)

data_cluster['clusters'] = y_kmeans

fig, ax = plt.subplots(figsize = (10, 8))

sns.scatterplot(ax = ax, data = score_science, x = data_cluster[data_cluster['clusters'] == 0]['avg_score_first_choice']

                , y = data_cluster[data_cluster['clusters'] == 0]['avg_score'], color = 'gray', label = 'Uncertainty')

sns.scatterplot(ax = ax, data = score_science, x = data_cluster[data_cluster['clusters'] == 1]['avg_score_first_choice']

                , y = data_cluster[data_cluster['clusters'] == 1]['avg_score'], color = 'green', label = 'Likely to Be Accepted')

sns.scatterplot(ax = ax, data = score_science, x = data_cluster[data_cluster['clusters'] == 2]['avg_score_first_choice']

                , y = data_cluster[data_cluster['clusters'] == 2]['avg_score'], color = 'red', label = 'Likely to Be Rejected')

sns.scatterplot(ax = ax, x = kmeans.cluster_centers_[:,1], y = kmeans.cluster_centers_[:,0], s= 200, color= 'black', label = 'Centroids')

ax.set_title('Clusters of Participants by Their Average Scores (Science Majors)', fontsize = 18)

ax.set_xlabel('Average Scores of Selected Majors and Universities')

ax.set_ylabel('Participants Average Scores')

plt.show()
data_cluster.loc[data_cluster['clusters'] == 0, 'clusters_information'] = 'Uncertainty'

data_cluster.loc[data_cluster['clusters'] == 1, 'clusters_information'] = 'Likely to Be Accepted'

data_cluster.loc[data_cluster['clusters'] == 2, 'clusters_information'] = 'Likely to Be Rejected'

cluster_first_choice_science = data_cluster.copy()

cluster_first_choice_science.head()
from sklearn.cluster import KMeans



data_cluster = score_science[['id_user', 'specific_second_choice', 'avg_score', 'avg_score_second_choice', ]]

X = data_cluster.iloc[:, 2:4].values

# I choose numbers of cluster based on the Elbow Method

kmeans = KMeans(n_clusters = 3, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)

y_kmeans = kmeans.fit(X)

y_kmeans = kmeans.predict(X)

data_cluster['clusters'] = y_kmeans

fig, ax = plt.subplots(figsize = (10, 8))

sns.scatterplot(ax = ax, data = score_science, x = data_cluster[data_cluster['clusters'] == 0]['avg_score_second_choice']

                , y = data_cluster[data_cluster['clusters'] == 0]['avg_score'], color = 'gray', label = 'Uncertainty')

sns.scatterplot(ax = ax, data = score_science, x = data_cluster[data_cluster['clusters'] == 1]['avg_score_second_choice']

                , y = data_cluster[data_cluster['clusters'] == 1]['avg_score'], color = 'red', label = 'Likely to Be Rejected')

sns.scatterplot(ax = ax, data = score_science, x = data_cluster[data_cluster['clusters'] == 2]['avg_score_second_choice']

                , y = data_cluster[data_cluster['clusters'] == 2]['avg_score'], color = 'green', label = 'Likely to Be Accepted')

sns.scatterplot(ax = ax, x = kmeans.cluster_centers_[:,1], y = kmeans.cluster_centers_[:,0], s= 200, color= 'black', label = 'Centroids')

ax.set_title('Clusters of Participants by Their Average Scores (Science Majors)', fontsize = 18)

ax.set_xlabel('Average Scores of Selected Majors and Universities')

ax.set_ylabel('Participants Average Scores')

plt.show()
data_cluster.loc[data_cluster['clusters'] == 0, 'clusters_information'] = 'Uncertainty'

data_cluster.loc[data_cluster['clusters'] == 1, 'clusters_information'] = 'Likely to Be Rejected'

data_cluster.loc[data_cluster['clusters'] == 2, 'clusters_information'] = 'Likely to Be Accepted'

cluster_second_choice_science = data_cluster.copy()

cluster_second_choice_science.head()