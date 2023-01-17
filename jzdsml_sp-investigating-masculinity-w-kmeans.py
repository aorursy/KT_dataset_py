import pandas as pd

survey = pd.read_csv("../input/masculinity.csv")

survey.head(2)
survey.isna().any()
print(survey.columns)

print(len(survey))

print("multiple parts -> multiple columns with each entry indicating personal choice for each part")

print(len(survey[survey.q0007_0001=='Often']))

print(survey.q0007_0001.value_counts())
cols_to_map = ["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004",

       "q0007_0005", "q0007_0006", "q0007_0007", "q0007_0008", "q0007_0009",

       "q0007_0010", "q0007_0011"]



# These questions are all related to frequency. We can map to numerical values.

choices = [[] for i in range(len(cols_to_map))]

mappings = [{} for i in range(len(cols_to_map))]

for i in range(len(cols_to_map)):

    #choices[i]= list(survey[cols_to_map[i]].value_counts().index.unique()) # order is not correct

    choices[i] = ['Sometimes',

                  'Rarely',

                  'Often',

                  'Never, but open to it',

                  'Never, and not open to it',

                  'No answer']

    mappings[i]= dict(zip(choices[i], list(range(len(choices[i])-1,-1,-1))))

    survey[cols_to_map[i]] = survey[cols_to_map[i]].map(mappings[i])

survey.head(20)
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(12,7))

ax = plt.subplot(1,1,1)

plt.scatter(survey["q0007_0001"],survey["q0007_0002"],alpha=0.1)

ax.set_xticks(range(len(choices[0])-1,-1,-1))

ax.set_yticks(range(len(choices[1])-1,-1,-1))

ax.set_xticklabels(choices[0][::-1])

ax.set_yticklabels(choices[1][::-1])

plt.xlabel('How often do you ask a friend for professional advice?')

plt.ylabel('How often do you ask a friend for personal advice?')

from sklearn.cluster import KMeans

print(len(survey))

rows_to_cluster = survey.dropna(subset=cols_to_map[0:5] + cols_to_map[7:9])

data_to_cluster = rows_to_cluster[cols_to_map[0:5] + cols_to_map[7:9]]

data_to_cluster = data_to_cluster[(data_to_cluster!=0).all(axis=1)]

data_to_cluster['not_masc'] = (data_to_cluster[cols_to_map[0]] + data_to_cluster[cols_to_map[1]] +\

                               data_to_cluster[cols_to_map[2]] + data_to_cluster[cols_to_map[3]])/4

data_to_cluster['masc'] = (data_to_cluster[cols_to_map[4]] + data_to_cluster[cols_to_map[7]] +\

                               data_to_cluster[cols_to_map[8]])/4

#0 => no answer. drop 0.

print(len(data_to_cluster))

classifier = KMeans(n_clusters=2)

classifier.fit(data_to_cluster[cols_to_map[0:5] + cols_to_map[7:9]])

print(classifier.cluster_centers_)

print(classifier.labels_)

cluster_zero_indices = [i for i in range(len(classifier.labels_)) if classifier.labels_[i]==0]

cluster_one_indices = [i for i in range(len(classifier.labels_)) if classifier.labels_[i]==1]

print(cluster_zero_indices)

cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]

cluster_one_df = rows_to_cluster.iloc[cluster_one_indices]

print(cluster_zero_df.educ4.value_counts()/len(cluster_zero_df))

print(cluster_one_df.educ4.value_counts()/len(cluster_one_df))

print(cluster_zero_df.age3.value_counts()/len(cluster_zero_df))

print(cluster_one_df.age3.value_counts()/len(cluster_one_df))



import numpy as np



figure = plt.figure(figsize=(12,7))

ax = plt.subplot(1,1,1)

plt.scatter(data_to_cluster['not_masc'], data_to_cluster['masc'], c=classifier.labels_, alpha=0.2, s=50, cmap='cool') #cool: 0-1 => green to red

plt.xlabel('Average of frequency of \"non-masculine\" activities')

plt.ylabel('Average of frequency of \"masculine\" activities')



#centers = ...

#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);



#The figure suggest that the first 4 questions in Part 7 is more significant in masculinity clustering than questions 589.



#If you have time you can even plot this clustering figure for every 2-question combination. E.g. question 3 and 4
#Check question 3 and 4



figure = plt.figure(figsize=(12,7))

ax = plt.subplot(1,1,1)

plt.scatter(data_to_cluster[cols_to_map[2]], data_to_cluster[cols_to_map[3]], c=classifier.labels_, alpha=0.2, s=50, cmap='cool') #cool: 0-1 => green to red

plt.xlabel('Question 3')

plt.ylabel('Question 4')

#Somewhat diagonal separation



#Question 3 performs best in classifying masculinity.