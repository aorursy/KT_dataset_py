import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

import math



pd.options.display.max_columns = None
!ls ../input/
data = pd.read_csv('../input/bank-additional-full.csv', delimiter=';')

data.tail()
'''

Renaming columns just for better understanding

'''



data.rename(columns={'housing':'housing_loan'}, inplace=True)
data.info()
data.dtypes
'''

The column duration is not supposed to be used, since we only know the call duration

after it has been finished, so it is considered data leakage

'''



data.drop(['duration'], axis=1, inplace=True)
catg_cols = ['job', 'marital', 'education', 'default', 'housing_loan',

             'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous',

             'poutcome']





num_cols = ['age', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',

            'euribor3m', 'nr.employed']
data.describe()
'''

I just want to see how many of each feature there is in each column



'''





unique_df = pd.DataFrame()



index=0

for column in catg_cols:

    

    unique_df = pd.concat([unique_df,pd.DataFrame(data[column].value_counts()).reset_index()], axis=1)

    unique_df[index] = '........'

    index+=1



unique_df.fillna('')
def plot_bar(col, size=(15,8), hue=True):

    fig, ax = plt.subplots()

    sns.set_style("whitegrid")

    if hue:

        sns.countplot(col, hue='y', data=data)

    else:

        sns.countplot(col, data=data)

        

    fig.set_size_inches(size)

    plt.xlabel(col) # Set text for the x axis

    plt.ylabel('Count')# Set text for y axis

    plt.show()

    
plot_bar('y', hue=False)
plot_bar('campaign')
'''

I'm gonna group more than 5 contacts all into one category

'''



data['campaign'] = data['campaign'].apply(lambda x: x if x<5 else 5) # More than 5 contacts are grouped
plot_bar('campaign')
plot_bar('pdays')
sns.countplot(data[data['pdays']!=999]['pdays'], hue=data['y'])
'''

Going to group them into 3 categories



1 = equal or less than 10 days



2 = more than 10 days



0 = not contacted before



'''





def treat_pdays(value):

    

    if value <= 10:

        return 1

    if value > 10 and value <= 27:

        return 2

    if value > 27:

        return 0



data['pdays'] = data['pdays'].apply(treat_pdays)
plot_bar('pdays')
plot_bar('previous')
'''

Going to group them into 3 categories



1 = contacted once before



2 = contacted more than once before



0 = not contacted before



'''





def treat_previous(value):

    

    if value == 0:

        return 0

    if value == 1:

        return 1

    else:

        return 2
data['previous'] = data['previous'].apply(treat_previous)
plot_bar('previous')
plot_bar('job')
'''

Merging housemaid into serices

'''



data['job'] = data['job'].replace('housemaid', 'services')
plot_bar('job')
'''

Getting dummies for the categorical columns

'''





dummy_features = pd.get_dummies(data[catg_cols])



num_features = data[num_cols]



print(dummy_features.shape)

print(num_features.shape)
'''

Scaling the numerical variables



'''



scaler = StandardScaler()



num_features = pd.DataFrame(scaler.fit_transform(num_features), columns=num_features.columns)
'''

Concatenating the scaled numerical columns with

the dummy columns

'''





preprocessed_df = pd.concat([dummy_features, num_features], axis=1)

preprocessed_df.shape
'''

Binarizing 'yes' and 'no'

values in the labels

'''



labels = data['y'].map({'no':0, 'yes':1})
pca = PCA(n_components=2)

pcs = pca.fit_transform(preprocessed_df)



pcs_df = pd.DataFrame(pcs)
def plot_2d(X, y, label='Classes'):   

 

    for l in zip(np.unique(y)):

        plt.scatter(X[y==l, 0], X[y==l, 1], label=l)

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()
plot_2d(pcs, labels)
from sklearn.cluster import KMeans
'''

Manually setting initial cluster centers

for improved accuracy

'''

n_clusters = 15 # 15 clusters from visual inspection above



cluster_centers = np.array([[-1.5,-1.5], [-1.6,-0.5], [-1.7,0.5], [-1.9,1.5], [-2,2.5],

                            [0.5,-1], [0,0], [0,1], [-0.2,2], [-0.5, 2.8],

                            [3,-1], [3,0], [2.5,1.1], [2.5,2], [2.5,3.2]])
kmeans = KMeans(n_clusters=n_clusters, max_iter=10000, verbose=1, n_jobs=4, init=cluster_centers)



clusters = kmeans.fit_predict(pcs_df)
pcs_df['cluster'] = clusters
'''

We can see that the clustering is acceptable

'''





plt.scatter(pcs_df[0], pcs_df[1], c=pcs_df['cluster'])
labels.value_counts()
'''

I'll extract 309 samples from each cluster

'''



n_samples = labels.value_counts()[1]//15

n_samples
'''

I'll select 309 random points from each cluster

But im only sampling the majority or label == 0



'''





index_list = []



for i in range(0,n_clusters):

    

    choices = pcs_df[(labels==0) & (pcs_df['cluster'] == i)].index

    

   

    

    index_list.append(np.random.choice(choices, n_samples))



    

index_list = np.ravel(index_list)
'''

Creating a new Dataframe with all the samples from the index_list which are all from the majority class

and all the samples from the minority class

'''



resampled_raw_data = pd.concat([data.iloc[index_list], data[data['y'] == 'yes']])
'''

Confirming concatenation

'''



resampled_raw_data.shape
'''

Confirming class imbalance

'''



resampled_raw_data['y'].value_counts()
'''

Saving resampled Dataframe for future classification task

'''





resampled_raw_data.to_csv('resampled_bank_data.csv')