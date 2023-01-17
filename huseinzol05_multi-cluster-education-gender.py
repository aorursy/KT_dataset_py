import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



# import seaborn for nice visualization, I am not very much like seaborn, too abstract

import seaborn as sns



# settings for matplotlib

sns.set(style="whitegrid", palette="muted")

current_palette = sns.color_palette()



# change string into int

from sklearn.preprocessing import LabelEncoder



# standardize into unit-variance, each value divide by variance

from sklearn.preprocessing import StandardScaler



# 'spring' related data into low dimension

from sklearn.manifold import TSNE



xrange = range
dataset = pd.read_csv('../input/Loan payments data.csv')

dataset.head()
#remove ID, dates, past_due_days

columns = ['Loan_ID', 'effective_date', 'due_date', 'paid_off_time', 'past_due_days']

for i in columns:

    del dataset[i]

dataset.head()
dataset_copy = dataset.copy()



label_loan_status = np.unique(dataset.ix[:, 0].values)



# change strings value into int, sorted by characters

for i in xrange(dataset_copy.ix[:, :].shape[1]):

    if str(type(dataset_copy.ix[0, i])).find('str') > 0:

        dataset_copy.ix[:, i] = LabelEncoder().fit_transform(dataset_copy.ix[:, i])



# first column as cluster classes

Y = dataset_copy.ix[:, 0].values



X = dataset_copy.ix[:, 1:].values



X = StandardScaler().fit_transform(X)

X = TSNE(n_components = 2).fit_transform(X)



for no, _ in enumerate(np.unique(Y)):

    plt.scatter(X[Y == no, 0], X[Y == no, 1], color = current_palette[no], label = label_loan_status[no])

    

plt.legend()

plt.show()
dataset_copy = dataset.copy()



label_loan_status = np.unique(dataset.ix[:, 0].values)

label_education = np.unique(dataset.ix[:, 4].values)



columns = ['Principal', 'terms']



for i in xrange(dataset_copy.ix[:, :].shape[1]):

    if str(type(dataset_copy.ix[0, i])).find('str') > 0:

        dataset_copy.ix[:, i] = LabelEncoder().fit_transform(dataset_copy.ix[:, i])



fig = plt.figure(figsize=(10,15))



Y = dataset['age'].ix[:].values



labelset = dataset_copy['loan_status'].ix[:].values



num = 1



for i in xrange(len(label_education)):

    for k in xrange(len(columns)):

        

        plt.subplot(len(label_education), len(columns), num)

        

        X = dataset_copy[columns[k]].ix[:].values

        

        X = X[dataset_copy['education'].ix[:].values == i]

        

        Y_in = Y[dataset_copy['education'].ix[:].values == i]

           

        labelset_filter = labelset[dataset_copy['education'].ix[:].values == i]

        

        for no, text in enumerate(label_loan_status):

            plt.scatter(X[labelset_filter == no], Y_in[labelset_filter == no], label = text, color = current_palette[no])

        

        plt.legend()

        plt.xlabel(columns[k])

        plt.ylabel('Age')

        plt.title(label_education[i])

        

        num += 1



fig.tight_layout()        

plt.show() 
dataset_copy = dataset.copy()



label_loan_status = np.unique(dataset.ix[:, 0].values)

label_gender = np.unique(dataset.ix[:, 5].values)



columns = ['Principal', 'terms']



for i in xrange(dataset_copy.ix[:, :].shape[1]):

    if str(type(dataset_copy.ix[0, i])).find('str') > 0:

        dataset_copy.ix[:, i] = LabelEncoder().fit_transform(dataset_copy.ix[:, i])



fig = plt.figure(figsize=(10,10))



Y = dataset['age'].ix[:].values



labelset = dataset_copy['loan_status'].ix[:].values



num = 1



for i in xrange(len(label_gender)):

    for k in xrange(len(columns)):

        

        plt.subplot(len(label_gender), len(columns), num)

        

        X = dataset_copy[columns[k]].ix[:].values

        

        X = X[dataset_copy['Gender'].ix[:].values == i]

        

        Y_in = Y[dataset_copy['Gender'].ix[:].values == i]

           

        labelset_filter = labelset[dataset_copy['Gender'].ix[:].values == i]

        

        for no, text in enumerate(label_loan_status):

            plt.scatter(X[labelset_filter == no], Y_in[labelset_filter == no], label = text, color = current_palette[no])

        

        plt.legend()

        plt.xlabel(columns[k])

        plt.ylabel('Age')

        plt.title(label_education[i])

        

        num += 1



fig.tight_layout()        

plt.show() 