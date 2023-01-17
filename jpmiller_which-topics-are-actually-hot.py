import numpy as np

import pandas as pd



train = pd.read_csv('../input/train.csv')

train.tail()
trainstay = train.loc[train['is_duplicate'] == 1, ['qid1', 'qid2']]

stays = pd.Series(trainstay.values.ravel()).unique().tolist()

allvals = list(range(1, 537934)) # one larger than our max qid

solos = set(allvals) - set (stays)

print(len(solos))

print(max(allvals))
qid1 = trainstay['qid1'].tolist()

qid2 = trainstay['qid2'].tolist()

mypairs = list(zip(qid1, qid2))

mypairs[0:10]
def connected_tuples(pairs):

    # for every element, we keep a reference to the list it belongs to

    lists_by_element = {}



    def make_new_list_for(x, y):

        lists_by_element[x] = lists_by_element[y] = [x, y]



    def add_element_to_list(lst, el):

        lst.append(el)

        lists_by_element[el] = lst



    def merge_lists(lst1, lst2):

        merged_list = lst1 + lst2

        for el in merged_list:

            lists_by_element[el] = merged_list



    for x, y in pairs:

        xList = lists_by_element.get(x)

        yList = lists_by_element.get(y)



        if not xList and not yList:

            make_new_list_for(x, y)



        if xList and not yList:

            add_element_to_list(xList, y)



        if yList and not xList:

            add_element_to_list(yList, x)            



        if xList and yList and xList != yList:

            merge_lists(xList, yList)



    # return the unique lists present in the dictionary

    return set(tuple(l) for l in lists_by_element.values())





cpairs =  connected_tuples(mypairs)

print(list(cpairs)[0:30])
universe = cpairs.union(solos)

print ("Item count:", len(universe))
uni2 = list(universe)



ctlist = []

i = 0  

while i < len(uni2):  

  item = str(uni2[i])

  ct = item.count(',') + 1

  ctlist.append(ct)  

  i += 1 

print('Number of Questions in all Sets: {}'.format(sum(ctlist)))

print('Lengths of Connected Sets')



# put it in d dataframe

qSets = pd.DataFrame(

    {'qid': uni2,

    'set_length': ctlist}

    )

qSets.sort_values('set_length', axis=0, ascending=False, inplace=True)

qSets.reset_index(inplace=True, drop=True)

qSets['set_id'] = qSets.index + 1

qSets.head()
# separate out the integers from the lists

qSetsS = qSets.loc[qSets['set_length'] == 1] 

qSetsL = qSets.loc[qSets['set_length'] > 1] 



# unnest

rows = []

_ = qSetsL.apply(lambda row: [rows.append([row['set_id'], row['set_length'], nn]) 

                         for nn in row.qid], axis=1)



qRef = pd.DataFrame(rows, columns = ['set_id', 'set_length', 'qid'])



qRef = qRef.append(qSetsS)

qRef.sort_values('qid', inplace=True)

qRef.reset_index(inplace=True, drop=True)

qRef.to_csv('qRef.csv', index=False)

qRef.head()
# create a lookup table from train

q1s = train.iloc[:, [1,3]]

q2s = train.iloc[:, [2,4]]



new_cols = ['qid', 'question']

q1s.columns = new_cols

q2s.columns = new_cols



lookup = pd.concat([q1s, q2s], ignore_index=True)

lookup.drop_duplicates('qid', inplace=True)
qTop = qRef.drop_duplicates('set_id', keep='first')

j = qTop.merge(lookup, how='left', on='qid')

j.sort_values('set_length', ascending=False).head(6)