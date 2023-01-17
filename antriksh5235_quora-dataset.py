# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



pal = sns.color_palette()



print('# File sizes')

for f in os.listdir('../input'):

    if 'zip' not in f:

        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')



# Any results you write to the current directory are saved as output.
def length(sent):

    words = str(sent).split()

    return len(words)
train = pd.read_csv('../input/train.csv')

main = []

for item in train.itertuples():

    main.append((length(str(item[4])), length(str(item[5]))))

#    break

print(len(main))



train = pd.DataFrame(main, columns=['len_q1','len_q2'])

train.head()
#Question 2 Lengths

plt.hist(train['len_q2'], bins=50, range=[0,50])

plt.show()

plt.close()

#Question 1 Lengths

plt.hist(train['len_q1'], bins=50, range=[0,50])

plt.show()