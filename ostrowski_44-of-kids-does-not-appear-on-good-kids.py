# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

from collections import Counter # advanced counting

from pylab import rcParams # advanced time series visualization





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Loading data as numpy arrays using np.genfromtxt()

child_wishlist = np.genfromtxt('../input/child_wishlist.csv', delimiter=',', dtype='int64')

gift_goodkids = np.genfromtxt('../input/gift_goodkids.csv', delimiter=',', dtype='int64')
# Be careful to not misinterpret child_wishlist and gift_goodkids! Read documentation!



# Checking the .shape of child_wishlist, it is 11 since index 0 is kid_id followed by 10 gift_id

print(child_wishlist[1])

print(child_wishlist.shape, '\n')



# Checking the .shape of gift_goodkids, it is 1001 since index 0 is kid_id followed by 1000 gift_id

print(gift_goodkids[0])

print(gift_goodkids.shape)
# Getting a new_wishlist without kid_id

new_wishlist = []

for kid in child_wishlist:

    new_wishlist += list(kid[1:])
# .plot() the histogram of gift_frequencies

rcParams['figure.figsize'] = 11, 9

plt.hist(new_wishlist, bins=1000)

plt.title("Distribution of gift frequencies")

plt.xlabel("Gift number")

plt.ylabel("Frequency")

plt.show()
# Getting 10 loved_gift and the top 10 hated

loved_gift = Counter(new_wishlist).most_common()

print('10 top most loved presents')

for gift in loved_gift[:10]:

    print('Gift ' + str(gift[0]) + ' : ' + str(gift[1]) + ' times')



print('\n10 hated presents')

for gift in loved_gift[-10:-1]:

    print('Gift ' + str(gift[0]) + ' : ' + str(gift[1]) + ' times')    
# Getting a g_kids list

g_kids = []

for kid in gift_goodkids:

    g_kids += list(kid[1:])



# Getting 10 loved_gift and the top 10 hated

cg_kids = Counter(g_kids).most_common()



print('10 top most lovely kids')

for kid in cg_kids[:10]:

    print('Kid ' + str(kid[0]) + ' : ' + str(kid[1]) + ' times')

    

print('\n10 bad behaved kids')

for kid in cg_kids[-10:-1]:

    print('Kid ' + str(kid[0]) + ' : ' + str(kid[1]) + ' times')
# Total amount of kids that does not appear in gift_goodkids list

1 - len(np.unique(g_kids))/1E6



# TODO generate a mask with children that does not appear on the gift_goodkids list
# Generating a mask of children not in the gift_goodkids

kids_in_ggk = np.unique(g_kids)

mask = np.zeros(1000000)



for value in (kids_in_ggk):

    mask[value] = value

    

mask_bl = np.where(mask != 0, True, False)



#Checking if .len() and .sum() are the same of previos evaluations

print(mask_bl)

print(len(mask_bl))

print(mask_bl.sum())