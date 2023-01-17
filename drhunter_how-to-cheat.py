import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from collections import defaultdict



def get_sequences(filename='oeis'):

    with open(filename, 'r') as f:

        for l in f:

            yield l.rstrip(',\n').split(',')



sequences = {s[0]: s[1:] for s in get_sequences()}



def tree():

    return defaultdict(tree)



# This uses each entry in a sequence as keys to traverse the tree

def traverse(s, t):

    if not s:

        return t

    return traverse(s[1:], t[s[0]])



sequence_tree = tree()



# Because the tree is made of defaultdicts, traversing the tree builds it

for name, seq in sequences.iteritems():

    node = traverse(seq, sequence_tree)

    node['name'] = name # Might as well



test_data = pd.read_csv('test.csv')





results = [(traverse(row.Sequence.split(','), sequence_tree).keys() + [0])[0] for _, row in test_data.iterrows()]



submission = pd.DataFrame(results, test_data.Id, ['Last'])

submission.index.name = 'Id'