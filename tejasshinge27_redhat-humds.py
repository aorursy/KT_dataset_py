# importing basic libraries

import pandas as pd

from pandas import Series,DataFrame



# importing libraries we need for analysis and visualizations



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# File 1 People.CSV file

# Description -- The people file contains all of the unique people (and the corresponding characteristics) that have 

#performed activities over time Each row in the people file represents a unique person. 

#Each person has a unique people_id.



people=pd.read_csv('../input/people.csv')



#People Data Cleanup



for i in list(people.columns):

    if i not in ['people_id', 'activity_id', 'date', 'char_38', 'outcome']:

        if people[i].dtype == 'object':

            people[i].fillna('type 0', inplace=True)

            people[i] = people[i].map(lambda x: x.split(' ')[1]).astype(np.int32)

        elif people[i].dtype == 'bool' :

            people[i] = people[i].astype(np.int8)



people.head()

#people.info()















#File 4 act_train.csv



act_train=pd.read_csv('../input/act_train.csv')

act_train.head()

act_train.info()

merge_ds=pd.merge(people,act_train, how='left', on=['people_id'])

merge_ds.head()

merge_ds.info()
