# Import Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from kmodes.kmodes import KModes

from sklearn import preprocessing

from sklearn.decomposition import PCA



pd.set_option('mode.chained_assignment', None)
# Define Methods



def transform_true_false(x):

    if x == 'T':

        return 1

    elif x == 'F':

        return 0

    else:

        return -1

    

def transform_yes_no(x):

    if x == 'Y':

        return 1

    elif x == 'N':

        return 0

    else:

        return -1

    

def transform_ord_0(x):

    if x == 1.0:

        return 0

    elif x == 2.0:

        return 1

    elif x == 3.0:

        return 2

    else:

        return -1

    

def transform_ord_1(x):

    if x == 'Novice':

        return 0

    elif x == 'Contributor':

        return 1

    elif x == 'Expert':

        return 2

    elif x == 'Master':

        return 3

    elif x == 'Grandmaster':

        return 4

    else:

        return -1

    

def transform_ord_2(x):

    if x == 'Freezing':

        return 0

    elif x == 'Cold':

        return 1

    elif x == 'Warm':

        return 2

    elif x == 'Hot':

        return 3

    elif x == 'Boiling Hot':

        return 4

    elif x == 'Lava Hot':

        return 5

    else:

        return -1   

    

ord_3_dict = { 'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14 }

ord_4_dict = { 'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25 }
directory = "../input/cat-in-the-dat-ii/"

feature_tables = ['train.csv','test.csv','sample_submission.csv']



df_train = directory + feature_tables[0]

df_test = directory + feature_tables[1]

sample_submission = directory + feature_tables[2]



# Create dataframes

print(f'Reading csv from {df_train}...')

train = pd.read_csv(df_train)

print('...Complete')



print(f'Reading csv from {df_test}...')

test = pd.read_csv(df_test)

print('...Complete')



print(f'Reading csv from {sample_submission}...')

sample_submission = pd.read_csv(sample_submission)

print('...Complete')
train.head()
bin_features = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

nom_features = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

ord_features = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4']

time_features = ['day', 'month']

target = ['target']



all_features = bin_features + nom_features + ord_features + time_features + target
for col in all_features:

    print(f'Filling in missing catagorical data in {col} with mode of {col}...')

    train[col].fillna(train[col].mode()[0], inplace = True)

    

    print(f'Number of unique catagories in {col} = {train[col].nunique()}\n')
train_bin = train[bin_features]

train_bin['bin_3'] = train_bin['bin_3'].apply(transform_true_false)

train_bin['bin_4'] = train_bin['bin_4'].apply(transform_yes_no)

train_bin = train_bin.astype('int64')

train_bin.head()
train_nom = train[nom_features]

for col in nom_features:

    le = preprocessing.LabelEncoder()

    train_nom[col] = le.fit_transform(train_nom[col])

    

train_nom.head()
train_ord = train[ord_features]

train_ord['ord_0'] = train_ord['ord_0'].apply(transform_ord_0)

train_ord['ord_1'] = train_ord['ord_1'].apply(transform_ord_1)

train_ord['ord_2'] = train_ord['ord_2'].apply(transform_ord_2)

train_ord['ord_3'] = train_ord['ord_3'].map(ord_3_dict)

train_ord['ord_4'] = train_ord['ord_4'].map(ord_4_dict)



train_ord.head()
train_time = train[time_features]

train_time['day'] = train_time['day'].apply(lambda x: x-1)

train_time['month'] = train_time['month'].apply(lambda x: x-1)

train_time = train_time.astype('int64')



train_time.head()
train_final = pd.concat([train_bin, train_nom, train_ord, train_time, train[target]], axis = 1)

train_final.head()
corr = train_final.corr(method='spearman')

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(20, 18))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap="YlGnBu", vmax=.30, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
cost = []

K = range(1,5)

for num_clusters in list(K):

    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)

    kmode.fit_predict(train_final)

    cost.append(kmode.cost_)

    

plt.plot(K, cost, 'bx-')

plt.xlabel('k clusters')

plt.ylabel('Cost')

plt.title('Elbow Method For Optimal k')

plt.show()
km = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)

cluster_labels = km.fit_predict(train_final)

train['Cluster'] = cluster_labels
for col in all_features:

    plt.subplots(figsize = (15,5))

    sns.countplot(x='Cluster',hue=col, data = train)

    plt.show()