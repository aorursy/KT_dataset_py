# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Libraries

import pandas as pd

import numpy as np

import os

import seaborn as sns

import matplotlib.pyplot as plt
# Importing data

df = pd.read_csv('../input/unbalancedrisk/Sample_Dataset.csv')

df.head()
# Quantity of items 0 and 1

cnt = len(df['obj1'])

cnt
# Record count equal to 0

cnt_obj_0 = len(df[df['obj1']==0])

cnt_obj_0
# Record count equal to 1

cnt_obj_1 = len(df[df['obj1']==1])

cnt_obj_1
# How much more is 0 compared to 1

qtd_vezes_maior = '{:.2f}'.format(cnt_obj_0/cnt_obj_1)

print('The quantity of 0 is '+str(qtd_vezes_maior)+' times greater than the quantity of 1')
# Setting the background for the chart

sns.set_style('darkgrid')



# Building the count graph

ax = sns.countplot(df['obj1'], palette="Set3")



# Percentage of 0 and 1

perc_0 = '{:.2f}%'.format(100*(cnt_obj_0/cnt))

perc_1 = '{:.2f}%'.format(100*(cnt_obj_1/cnt))



# Set the Xlabel

plt.xlabel('Object')



# Inserting the percentages on the chart

plt.text(1,3000,"% of 0: "+perc_0+"\n% of 1: "+perc_1)



# Show the graph

plt.show()
# Shuffle the data before creating the subsets

df = df.sample(frac=1)



# Defining the size of the range we want from the data [0 and 1]

# We need to use the loc function to access the dataset in the dataset from a starting and ending position

df_0 = df.loc[df['obj1']==0][:834]

df_1 = df.loc[df['obj1']==1]



# Concatenating the subsets

df2 = pd.concat([df_0,df_1])



# Shuffling the data

new_df = df2.sample(frac=1, random_state = 50)

new_df.head()
# Checking the new distribution of records equal to 0 and 1

print(new_df['obj1'].value_counts()/len(new_df))
# Get the number of occurrences in our new dataset

ncount = len(new_df['obj1'])



# Checking the balance on the chart

ax2 = sns.countplot(new_df['obj1'], palette="Set3")



# Make twin axis ax2

ax3 = ax2.twinx()



# Count on right, frequency on left

ax3.yaxis.tick_left()

ax2.yaxis.tick_right()



# The same thing to the labels

ax3.yaxis.set_label_position('left')

ax2.yaxis.set_label_position('right')



# set the frequency label

ax3.set_ylabel('Frequency (%)')



# Seting the % for each count [0, 1]

for p in ax2.patches:

    x=p.get_bbox().get_points()[:,0] #Get all of x locations

    y=p.get_bbox().get_points()[1,1] #Get all of y locations

    ax2.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

           ha='center', va='bottom')

    

# Fix the frequency range to 0-100 and the count range to ncount+1000

ax3.set_ylim(0,100)

ax2.set_ylim(0,ncount)



# turn the grid on ax2 off

ax3.grid(None)



#Show the plot

plt.show()