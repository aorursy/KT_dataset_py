from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

import numpy as np

import pandas as pd

import seaborn as sns
data = pd.read_csv('/kaggle/input/slogan-dataset/sloganlist.csv')
data
data.drop_duplicates(subset ="Company",keep = False, inplace = True) 
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data.Slogan)


def find_the_most_rep_slogan():

    global the_firm_list

    true_k = 3

    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

    model.fit(X)

    y = model.predict(X)

    data['categories'] = y

    for comp in the_firm_list:

        for element in data.loc[data['categories'] == data.categories.mode()[0]]['Company'].values:

            if element in the_firm_list:

                the_firm_list.remove(element)
the_most_original_firms = []

for i in range(3):

    the_List = []

    for j in range(30):

        the_firm_list = list(data.Company.unique())

        for k in range(15):

            find_the_most_rep_slogan()



        the_List += the_firm_list

        

    the_most_original_firms += list(set([i for i in the_List if the_List.count(i)>1]))
set([i for i in the_most_original_firms if the_most_original_firms.count(i)>1])
data