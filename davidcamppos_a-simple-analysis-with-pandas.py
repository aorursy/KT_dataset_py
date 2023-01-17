# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# reading one of the files

pd.read_excel("/kaggle/input/loto-fcil/loto_facil_2020_sorteio.xlsx").head(9)
# Files consolidation

all_files = glob.glob("/kaggle/input/loto-fcil/loto_facil_*.xlsx")



li = []



for filename in all_files:

    df = pd.read_excel(filename, index_col=None, header=0)

    header = df.iloc[5] 

    df = df[6:] 

    df = df.rename(columns = header)

    df = df.sort_index(ascending=False)

    df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y")

    li.append(df)

    

drawn_df = pd.concat(li, axis=0, ignore_index=True)
# read columns

drawn_df.columns
# Rename columns

drawn_df.columns = ['loto_facil', 'date', 'ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5',

                    'ball_6', 'ball_7', 'ball_8', 'ball_9', 'ball_10', 'ball_11', 'ball_12', 

                    'ball_13', 'ball_14', 'ball_15']

#Set multi index

drawn_df.set_index(['loto_facil', 'date'], inplace = True)
drawn_df = drawn_df.sort_index(ascending=True)
# read first and last line

drawn_df.loc[[1, len(drawn_df)]]
# read the last five line

drawn_df.tail()
# Describe dataframe

drawn_df.describe()
matrix = drawn_df.values

matrix
def bubble_sort(alist):

    for passnum in range(len(alist)-1, 0, -1):

        for i in range(passnum):

            if alist[i] > alist[i+1]:

                temp = alist[i]

                alist[i] = alist[i+1]

                alist[i+1] = temp





i = 0

while i < len(matrix):

    bubble_sort(matrix[i])

    i = i + 1
matrix
# Create dataframe

lottery_df = pd.DataFrame(matrix, index=drawn_df.index, 

                           columns=list(['num_1', 'num_2', 'num_3', 'num_4','num_5',

                                         'num_6', 'num_7','num_8', 'num_9', 'num_10',

                                         'num_11', 'num_12', 'num_13','num_14', 'num_15']))
# Read the last five line

lottery_df.tail()
lottery_df[lottery_df.duplicated(keep=False)]
# Describe dataframe

lottery_df.describe()
# value_counts()

lottery_df['num_1'].value_counts().to_frame()
# Filter data

lottery_df[lottery_df.num_1 == 6] # or => lottery_df[lottery_df['num_1'] == 6]
def array_into_2columns(alist):

    li = []

    for ind in range(len(alist)):

        for col in range(len(alist[0])):

            li.append([ind+1,alist[ind][col]])

            

    return li



two_columns = array_into_2columns(matrix)
two_columns_df = pd.DataFrame(two_columns, columns=['loto_facil','number'])

two_columns_df.set_index('loto_facil', inplace=True)

two_columns_df.head(31)
most_drawn = two_columns_df['number'].value_counts().to_frame(name="amount")

most_drawn