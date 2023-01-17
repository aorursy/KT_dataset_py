import pandas as pd



survey = pd.read_csv("../input/masculinity/masculinity.csv")

print('total rows:',survey.shape[0])

print('total columns:',survey.shape[1])

print('breakdown of column types:')

display(survey.dtypes.value_counts())

pd.set_option('display.max_columns', None)

display(survey.head(3))
summary = survey.nunique().sort_values(ascending=False).reset_index()

summary = summary[summary['index'].str.contains("q")]

print('total questions:',len(summary))

print('questions with more than 2 kind of answers:',len(summary[summary[0]>2]))

print('questions with only 2 kind of answers:',len(summary[summary[0]==2]))

display(summary[summary[0]>2],summary[summary[0]==2])
print(survey['q0007_0005'].unique())
cols_to_map = ["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004",

       "q0007_0005", "q0007_0006", "q0007_0007", "q0007_0008", "q0007_0009",

       "q0007_0010", "q0007_0011"]

count = 0

for i in cols_to_map:

    survey[i] = survey[i].map({'Often':4 ,'Sometimes':3,'Rarely':2,'Never, but open to it':1,'Never, and not open to it':0})

    count+=1

print('total',count,'columns mapped')
display(survey["q0007_0001"].unique())

display(survey["q0007_0002"].unique())

display(survey["q0007_0001"].head(5),survey["q0007_0002"].head(5))
questions=[['Ask a friend for professional advice'],

          ['Ask a friend for personal advice'],

['Express physical affection to male friends, like hugging, rubbing shoulders'],

['Cry'],

['Get in a physical fight with another person'],

['Have sexual relations with women, including anything from kissing to sex'],

['Have sexual relations with men, including anything from kissing to sex'],

['Watch sports of any kind'],

['Work out'],

['See a therapist'],

['Feel lonely or isolated']]
from matplotlib import pyplot as plt

import numpy as np

plt.style.use('bmh')

plt.figure(figsize=[15,12])

plt.subplot(2,2,1)

x=survey['q0007_0001']

y=survey['q0007_0002']

both_finite = np.isfinite(x) & np.isfinite(y)

plt.xlabel('professional advice')

plt.ylabel('personal advice')

m, b = np.polyfit(x[both_finite], y[both_finite], 1) #get best fit, 1 means linear.

plt.plot(x,m*x+b)

plt.scatter(x,y,alpha=0.06)



plt.subplot(2,2,2)

x=survey['q0007_0001']

y=survey['q0007_0003']

both_finite = np.isfinite(x) & np.isfinite(y)

plt.xlabel('professional advice')

plt.ylabel('Express physical affection to male friends, like hugging, rubbing shoulders')

m, b = np.polyfit(x[both_finite], y[both_finite], 1) #get best fit, 1 means linear.

plt.plot(x,m*x+b)

plt.scatter(x,y,alpha=0.06)



plt.subplot(2,2,3)

x=survey['q0007_0008']

y=survey['q0007_0006']

both_finite = np.isfinite(x) & np.isfinite(y)

plt.xlabel(questions[7])

plt.ylabel(questions[5])

m, b = np.polyfit(x[both_finite], y[both_finite], 1) #get best fit, 1 means linear.

plt.plot(x,m*x+b)

plt.scatter(x,y,alpha=0.06)



plt.subplot(2,2,4)

x=survey['q0007_0008']

y=survey['q0007_0006']

both_finite = np.isfinite(x) & np.isfinite(y)

plt.xlabel(questions[8])

plt.ylabel(questions[5])

m, b = np.polyfit(x[both_finite], y[both_finite], 1) #get best fit, 1 means linear.

plt.plot(x,m*x+b)

plt.scatter(x,y,alpha=0.06)
rows_to_cluster = survey.dropna(subset = ["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"])

print('initial rows:',survey[["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"]].shape[0])

print('drop nan rows...')

print('total rows:',rows_to_cluster.shape[0])

print('total columns:',rows_to_cluster.shape[1])



from sklearn.cluster import KMeans

model = KMeans(2,random_state=42)

model.fit(rows_to_cluster[["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"]])

print('The cluster centroids:\n',model.cluster_centers_)

centroids = model.cluster_centers_

centroids_list = []

for i in range(len(centroids)):

    centroids_list.append(list(centroids[i]))



x=range(len(["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"]))

for i in range(len(centroids_list)):

    y = centroids_list[i]

    plt.plot(x,y,label='cluster '+str(i))

plt.legend()
print(model.labels_)

print('total labels',len(model.labels_))

print('predicted labels\n',pd.Series(model.labels_).value_counts())

cluster_zero_indices = []

cluster_one_indices = []

for i in range(len(model.labels_)):

    if model.labels_[i] == 0: cluster_zero_indices.append(i)

    else: cluster_one_indices.append(i)

print(cluster_zero_indices)
# survey['educ4'] = survey['educ4'].map({'Post graduate degree':3,'College or more':2,'Some college':1,'High school or less':0})
cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]

cluster_one_df = rows_to_cluster.iloc[cluster_one_indices]
from scipy.stats import chi2_contingency

cols_compare = ['race2','racethn4', 'educ3', 'educ4', 'age3', 'kids', 'orientation']

outcome = []

significant_factors = []



print('\nAnalyze significant factors to the clusters:')

for i in cols_compare:

    x =list(zip(list(cluster_zero_df[i].value_counts()),list(cluster_one_df[i].value_counts())))

    chi2, pval, dof, expect = chi2_contingency(x)

    outcome.append([i,pval])

    if pval <= 0.05: significant_factors.append([i,pval]) 

    print([i,pval])

    

print('\nThe factors that contribute to the different clusters could be:')

for i in range(len(significant_factors)):

    print(significant_factors[i])
print('cluster 1 size:',len(cluster_zero_df))

print('cluster 2 size:',len(cluster_one_df))

print('\nDistribution of education:')

print(cluster_zero_df['educ4'].value_counts()/len(cluster_zero_df))

print(cluster_one_df['educ4'].value_counts()/len(cluster_one_df))
#check optimal clusters

min_ = 1

max_ = 10

inertia_ = []

cluster_centers_ = []

for i in range(min_,max_):

    model = KMeans(i,random_state=42)

    model.fit(rows_to_cluster[["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"]])

#     model.cluster_centers_

    inertia_.append(model.inertia_)

    

plt.figure(figsize=[13,5])

ax1 = plt.subplot(1,2,1)

plt.title('inertia of clusters used')

plt.plot(list(range(min_,max_)),inertia_)

plt.xlabel('n cluster')

plt.ylabel('inertia')
model = KMeans(3,random_state=42)

model.fit(rows_to_cluster[["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"]])

centroids = model.cluster_centers_

centroids_list = []

for i in range(len(centroids)):

    centroids_list.append(list(centroids[i]))



ax = plt.subplot(1,1,1)

x=range(len(["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"]))

name = ['A','B','C']

for i in range(len(centroids_list)):

    y = centroids_list[i]

    plt.plot(x,y,label='cluster'+str(name[i]))

ax.set_xticklabels(["","q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"], rotation =90)

plt.legend()
print(model.labels_)

print('total labels',len(model.labels_))

print('predicted labels\n',pd.Series(model.labels_).value_counts())

cluster_a_indices = []

cluster_b_indices = []

cluster_c_indices = []

for i in range(len(model.labels_)):

    if model.labels_[i] == 0: cluster_a_indices.append(i)

    elif model.labels_[i] == 1: cluster_b_indices.append(i)

    else: cluster_c_indices.append(i)



cluster_a_df = rows_to_cluster.iloc[cluster_a_indices]

cluster_b_df = rows_to_cluster.iloc[cluster_b_indices]

cluster_c_df = rows_to_cluster.iloc[cluster_c_indices]
cols_compare = ['race2','racethn4', 'educ3', 'educ4', 'age3', 'kids', 'orientation']

outcome = []

significant_factors = []



print('\nAnalyze significant factors to the clusters:')

for i in cols_compare:

    x =list(zip(list(cluster_a_df[i].value_counts()),list(cluster_b_df[i].value_counts()),list(cluster_c_df[i].value_counts())))

    print(x)

    chi2, pval, dof, expect = chi2_contingency(x)

    outcome.append([i,pval])

    if pval <= 0.05: significant_factors.append([i,pval]) 

    print([i,pval])

    

print('\nThe factors that contribute to the different clusters could be:')

for i in range(len(significant_factors)):

    print(significant_factors[i])
column = 'orientation'

print('cluster A size:',len(cluster_a_df))

print('cluster B size:',len(cluster_b_df))

print('cluster C size:',len(cluster_c_df))

print('\nDistribution of education:')

print('a\n',cluster_a_df[column].value_counts()/len(cluster_a_df))

print('b\n',cluster_b_df[column].value_counts()/len(cluster_b_df))

print('c\n',cluster_c_df[column].value_counts()/len(cluster_c_df))
#we do a survey hypothesis testing on education

from scipy.stats import chi2_contingency

X = [ [30, 10],

         [35, 15],

         [28, 12] ]

chi2, pval, dof, expect = chi2_contingency(X)
