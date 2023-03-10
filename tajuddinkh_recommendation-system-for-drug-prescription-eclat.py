# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import libraries

import plotly.offline as ply



ply.init_notebook_mode(connected=True)

%matplotlib inline



processed_data = pd.read_csv('/kaggle/input/drugs-prescriptions-with-providers/medicine_prescription_records.csv',header = 0).iloc[:,[0, 3]]

head_data = processed_data.head()
sample_data = processed_data.sample(n=50, random_state=1)
head_data
def process(df):

  df.columns = ['trans', 'item']

#   df.trans = df.trans.sample(n=20000, random_state=1)

#   df.item = df.item.sample(n=20000, random_state=1)

#   df.trans = pd.to_numeric(df.trans, errors='coerce')

#   df.trans = df.trans.astype(int)

  df.item = df.item.apply(lambda x: str(x).split(','))

  df = df.groupby('trans').agg(lambda x: x)

  return df



fit_data = process(head_data)
fit_data
# -*- coding: utf-8 -*-

"""

Eclat Algorithm

"""

import numpy as np

"""

The class is initialized with 3 parameters:

    min_support - is minimum support for an Item Set. min_support is taken as % of dataset length

    max_items - maximal number of elements in the Item Set

    min_items - minimal number of elements in the Item Set

"""

class Eclat:

    #initializing of the class object

    def __init__(self, min_support = 0.01, max_items = 5, min_items = 2):

        self.min_support = min_support

        self.max_items = max_items

        self.min_items = min_items

        self.item_lst = list()

        self.item_len = 0

        self.item_dict = dict()

        self.final_dict = dict()

        self.data_size = 0

    

    #creating a dicitionary of different non NA Items from all the trans-ns

    def read_data(self, dataset):

        for index, row in dataset.iterrows():

            row_wo_na = set(row[0])

            for item in row_wo_na:

                item = item.strip()

                if item in self.item_dict:

                    self.item_dict[item][0] += 1

                else:

                    self.item_dict.setdefault(item, []).append(1)

                self.item_dict[item].append(index)

        #set instance variables

        self.data_size = dataset.shape[0]

        self.item_lst = list(self.item_dict.keys())

        self.item_len = len(self.item_lst)

        self.min_support = self.min_support * self.data_size

        print ('Data read successfully')

        print ("min_supp", self.min_support)

        

    #recursive method to find all item sets in accordance with Eclat algorithm

    #data structure is the following: {Item: [Supp number, tid1, tid2, tid3, ...]}

    def recur_eclat(self, item_name, tids_array, minsupp, num_items, k_start):

        if tids_array[0] >= minsupp and num_items <= self.max_items:

            for k in range(k_start+1, self.item_len):

                if self.item_dict[self.item_lst[k]][0] >= minsupp:

                    new_item = item_name + " | " + self.item_lst[k]

                    new_tids = np.intersect1d(tids_array[1:], self.item_dict[self.item_lst[k]][1:])

                    new_tids_size = new_tids.size

                    new_tids = np.insert(new_tids, 0, new_tids_size)

                    if new_tids_size >= minsupp:

                        if num_items >= self.min_items: self.final_dict.update({new_item: new_tids})

                        self.recur_eclat(new_item, new_tids, minsupp, num_items+1, k)

    

    #call the described above functions for the given dataset

    def fit(self, dataset):

        i = 0

        self.read_data(dataset)

        for w in self.item_lst:

            self.recur_eclat(w, self.item_dict[w], self.min_support, 2, i)

            i+=1

        return self

        

    #output is a dictionary with ItemSets and absolute value of support

    def transform(self):

        return {k: "{0:.4f}%".format((v[0]+0.0)/self.data_size*100) for k, v in self.final_dict.items()}



# create an instance of the class with the required parameters

model = Eclat(min_support = 0.001, max_items = 4, min_items = 2)
%%time

# train with data

model.fit(fit_data)  
# and render the results

model.transform()