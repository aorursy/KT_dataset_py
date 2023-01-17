# references of code: https://tutorialedge.net/compsci/data-structures/getting-started-with-tries-in-python/



lists = []

def formatData(t,s):

    if not isinstance(t,dict) and not isinstance(t,list):

        print("\t"*s+str(t))

    else:

        for key in t:

            print("\t"*s+str(key))

            if not isinstance(t,list):

                formatData(t[key],s+1)



class Trie():



    def __init__(self, bound):

        self._end = '*'

        self.trie = dict()

        self.bound = bound

        

    def __repr__(self):

        return repr(self.trie)



    def add_word(self, word):

        temp_trie = self.trie

        

        for i in range(len(word)):

            if(i < self.bound):

#                 print(i, word[i])

                if word[i] in temp_trie:

                    temp_trie = temp_trie[word[i]]

                else:

                    temp_trie = temp_trie.setdefault(word[i], {})

            else:

                ref = temp_trie.get("<>")

                if not ref:

                    temp_trie['<>'] = []

                temp_trie['<>'].append(word[i:len(word)])

#                 temp_trie[self._end] = self._end

                break

#         temp_trie[self._end] = self._end

        return temp_trie



def count_bounds(trie):

    global lists

#     print(trie)   

    for sub_trie in trie:

        if sub_trie != '<>':

            count_bounds(trie[sub_trie])

        else:

#             print(trie[sub_trie], len(trie[sub_trie]))

            lists.append(len(trie[sub_trie]))



my_trie = Trie(3)

my_trie.add_word('head')

my_trie.add_word('heady')

my_trie.add_word('hi')

my_trie.add_word('howdy')

# print(my_trie)

formatData(my_trie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(my_trie.trie)

print('number of lists:', len(lists))

print('list size:', lists)



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

dfHist = pd.DataFrame.from_dict(data)



print(dfHist)

#plotting histogram

dfHist.plot.hist()
# dfImdb = pd.read_table('../input/imdb.txt', delim_whitespace=True, names=("W"))

dfImdb = pd.read_table('../input/imdb.txt', names=('W'), header=None)

print(dfImdb.shape)

dfImdb.head()
lists.clear()

imdbTrie3 = Trie(3)



for index, word in dfImdb.iterrows():

#     print(word['W'])

    imdbTrie3.add_word(str(word['W']))



# print(my_trie)

# formatData(imdb_trie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(imdbTrie3.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)



for l in lists:

    l = l / len(lists)

# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

                           ylim=[0.97,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0.97 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentual")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
lists.clear()

imdbTrie4 = Trie(4)



for index, word in dfImdb.iterrows():

#     print(word['W'])

    imdbTrie4.add_word(str(word['W']))



# print(my_trie)

# formatData(imdb_trie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(imdbTrie4.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)

# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

                           ylim=[0.98,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0.98 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
lists.clear()

imdbTrie5 = Trie(5)



for index, word in dfImdb.iterrows():

#     print(word['W'])

    imdbTrie5.add_word(str(word['W']))



# print(my_trie)

# formatData(imdb_trie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(imdbTrie5.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

                           ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0.99 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
lists.clear()

imdbTrie6 = Trie(6)



for index, word in dfImdb.iterrows():

#     print(word['W'])

    imdbTrie6.add_word(str(word['W']))



# print(my_trie)

# formatData(imdb_trie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(imdbTrie6.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

                           ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0.99 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
lists.clear()

imdbTrie7 = Trie(7)



for index, word in dfImdb.iterrows():

#     print(word['W'])

    imdbTrie7.add_word(str(word['W']))



# print(my_trie)

# formatData(imdb_trie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(imdbTrie7.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

                           ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0.99 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
lists.clear()

imdbTrie8 = Trie(8)



for index, word in dfImdb.iterrows():

#     print(word['W'])

    imdbTrie8.add_word(str(word['W']))



# print(my_trie)

# formatData(imdb_trie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(imdbTrie8.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

                           ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0.99 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
dfAol = pd.read_table('../input/aol.txt', names=('W'), header=None)



dfAol.shape
dfAol.head()
lists.clear()

aolTrie3 = Trie(3)



for index, word in dfAol.iterrows():

#     print(word['W'])

    aolTrie3.add_word(str(word['W']))



# print(aolTrie)

# formatData(aolTrie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(aolTrie3.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

#                            ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
lists.clear()

aolTrie4 = Trie(4)



for index, word in dfAol.iterrows():

#     print(word['W'])

    aolTrie4.add_word(str(word['W']))



# print(aolTrie)

# formatData(aolTrie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(aolTrie4.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)

# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

#                            ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
lists.clear()

aolTrie5 = Trie(5)



for index, word in dfAol.iterrows():

#     print(word['W'])

    aolTrie5.add_word(str(word['W']))



# print(aolTrie)

# formatData(aolTrie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(aolTrie5.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

#                            ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
lists.clear()

aolTrie6 = Trie(6)



for index, word in dfAol.iterrows():

#     print(word['W'])

    aolTrie6.add_word(str(word['W']))



# print(aolTrie)

# formatData(aolTrie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(aolTrie6.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

#                            ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
lists.clear()

aolTrie7 = Trie(7)



for index, word in dfAol.iterrows():

#     print(word['W'])

    aolTrie7.add_word(str(word['W']))



# print(aolTrie)

# formatData(aolTrie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(aolTrie7.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

#                            ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
lists.clear()

aolTrie8 = Trie(8)



for index, word in dfAol.iterrows():

#     print(word['W'])

    aolTrie8.add_word(str(word['W']))



# print(aolTrie)

# formatData(aolTrie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(aolTrie8.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

#                            ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
dfUsaddr = pd.read_table('../input/usaddr.txt', names=('W'), header=None)



dfUsaddr.shape
dfUsaddr.head()
lists.clear()

usaddr3 = Trie(3)



for index, word in dfUsaddr.iterrows():

#     print(word['W'])

    usaddr3.add_word(str(word['W']))



# print(aolTrie)

# formatData(aolTrie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(usaddr3.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

#                            ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
lists.clear()

usaddr4 = Trie(4)



for index, word in dfUsaddr.iterrows():

#     print(word['W'])

    usaddr4.add_word(str(word['W']))



# print(aolTrie)

# formatData(aolTrie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(usaddr4.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

#                            ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
lists.clear()

usaddr5 = Trie(5)



for index, word in dfUsaddr.iterrows():

#     print(word['W'])

    usaddr5.add_word(str(word['W']))



# print(aolTrie)

# formatData(aolTrie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(usaddr5.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

#                            ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
lists.clear()

usaddr6 = Trie(6)



for index, word in dfUsaddr.iterrows():

#     print(word['W'])

    usaddr6.add_word(str(word['W']))



# print(aolTrie)

# formatData(aolTrie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(usaddr6.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

#                            ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
lists.clear()

usaddr7 = Trie(7)



for index, word in dfUsaddr.iterrows():

#     print(word['W'])

    usaddr7.add_word(str(word['W']))



# print(aolTrie)

# formatData(aolTrie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(usaddr7.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

#                            ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)
lists.clear()

usaddr8 = Trie(8)



for index, word in dfUsaddr.iterrows():

#     print(word['W'])

    usaddr8.add_word(str(word['W']))



# print(aolTrie)

# formatData(aolTrie.trie,0)



print('########################')

print('\tResults:')

print('########################')



count_bounds(usaddr8.trie)

print('number of lists:', len(lists))



acc = 0

for l in lists:

    acc += l

print('average of list size:', acc/len(lists))

# print('list size:', lists)
# transforming array of list sizes into pandas dataframe

name = ['list_size']

data = {'list_size': lists}

bins= [1,5,10,50,100,1000,3000]

dfHist = pd.DataFrame.from_dict(data)



#plotting histogram 1

ax1 = dfHist.plot.hist(figsize=(12,5),fontsize=16)

ax1.set_xlabel('Size of Lists')

ax1.set_title('All frequency')



plt.show()



#plotting histogram 2

ax2 = dfHist.plot.hist(figsize=(12,5),

                           fontsize=16,

                           cumulative=True,

                           label='Size of list', 

#                            ylim=[7500,7755],

#                            ylim=[0.99,1],

#                            xlim=[0,3000],

                           histtype='step', # can be 'bar' or 'stopfilled' 

                           density=True #normalize data [0,1]

    )

ax2.set_xlabel('Size of Lists')

ax2.set_ylabel('Frequency normalized')

ax2.set_title('Frequency normalized bet 0 and 1')

plt.show()



#plotting Bars 3

out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

ax3 = out.value_counts(sort=False).plot.bar(

    rot=0, 

    color="y", 

    figsize=(12,5),

    fontsize=14

)

ax3.set_xlabel('Size of Lists')

ax3.set_ylabel('Frequency')

ax3.set_title('All frequency')

plt.show()







#plotting Bars 4

# out = pd.cut(dfHist['list_size'], bins=bins, include_lowest=True)

out_norm = out.value_counts(sort=False, normalize=True).mul(100)



ax4 = out_norm.plot.bar(

    rot=0, 

    color="c", 

    figsize=(16,7),

    fontsize=16

)

ax4.set_title('Percentage of Frequency')

plt.xlabel("size of each list")

plt.ylabel("percentage")

plt.show()



#print numeric of perc. for each interval

print("interval \t\t %")

print(out_norm)