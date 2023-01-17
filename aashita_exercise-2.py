import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.simplefilter('ignore')
path = '../input/titanic/'

train = pd.read_csv(path + 'train.csv') 

test = pd.read_csv(path + 'test.csv') 
df = pd.concat([train, test])
df.head()
df['Family'] = df['SibSp'] + df['Parch'] + 1

df.head()
sns.barplot(x='Family', y='Survived', data=df);
df['Ticket'].value_counts()[:15]
df.head()
df[df['GroupSize'] != df['Family']].shape[0], 

df[df['GroupSize'] != df['TicketCount']].shape[0]
path = '../input/competitive-data-science-predict-future-sales/'

df = pd.read_csv(path + 'sales_train.csv')

df.head()
df.head(20)
df.groupby(['date_block_num', 'item_id'])['item_cnt_day'].sum()
df.head(25)
df1 = pd.DataFrame({'CourseCode': ['PHYS024', 'CSCI35', 'ENGR156'], 

                   'CourseName': ['Mechanics and Wave Motion', 

                                  'Computer Science for Insight',

                                 'Intro to Comm & Info Theory']})



df2 = pd.DataFrame({'Professor': ['Zachary Dodds', 'Vatche Sahakian', 

                                  'Timothy Tsai', 'Brian Shuve'],

                    'CourseCode': ['CSCI35', 'PHYS024',  'ENGR156', 'PHYS024']})



df1.head()
df2.head()
pd.merge(df2, df1)
df.shape