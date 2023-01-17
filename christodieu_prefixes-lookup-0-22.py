import math



def findGCD(seq):

    gcd = seq[0]

    for i in range(1,len(seq)):

        gcd=math.gcd(gcd, seq[i])

    return gcd



print(findGCD([2,4,6,8]))
def findSignature(seq):

    nonzero_seq = [d for d in seq if d!=0]

    if len(nonzero_seq)==0:

        return seq

    sign = 1 if nonzero_seq[0]>0 else -1

    gcd = findGCD(seq)

    return [sign*x//gcd for x in seq]



print(findSignature([0,2,4,6,8]))
def findDerivative(seq):

    return [0] if len(seq)<=1 else [seq[i]-seq[i-1] for i in range(1,len(seq))]



print(findDerivative([1,1,2,3,5,8,13,21]))
def addAll(seq, node, list):

    if 'value' in node:

        list.append( ( seq, node['value'] ) )

    for key in node:

        if key != 'value':

            addAll(seq + [key], node[key], list)



class prefixTree:

    def __init__(self):

        self.data={}

        self.puts=0

        self.nodes=0

    

    def put(self, seq, value):

        node=self.data

        nodeCreated=False

        for i in range(0,len(seq)):

            item=seq[i]

            if not item in node:

                node[item]={}

                if 'value' in node:

                    del node['value']

                self.nodes+=1

                nodeCreated=True

            node=node[item]

        if nodeCreated:

            node['value']=value

            self.puts+=1

        elif 'value' in node:

            node['value']=max(node['value'], value)

    

    def prefix(self, seq):

        list=[]

        node=self.data

        for i in range(0,len(seq)):

            item=seq[i]

            if item in node:

                node=node[item]

            else:

                return list

        addAll(seq, node, list)

        return list

    

    def hasPrefix(self, seq):

        node=self.data

        for i in range(0,len(seq)):

            item=seq[i]

            if item in node:

                node=node[item]

            else:

                return False

        return True



sampleTrie=prefixTree()

sampleTrie.put([1,2,3], 50)

sampleTrie.put([1,2,4,9], 30)

sampleTrie.put([2,3,4], 20)

print(sampleTrie.prefix([1,2]))
"""

import datrie



class prefixTree:

    def __init__(self):

        self.data=datrie.Trie(',-0123456789')

        self.puts=0

        self.nodes=0

    

    def put(self, seq, value):

        key=','.join(map(str,seq))+','

        if key in self.data:

            self.data[key]=max(self.data[key],value)

        elif not self.data.has_keys_with_prefix(key):

            self.data[key]=value

            self.puts+=1

    

    def prefix(self, seq):

        ret=[]

        keys=self.data.keys(','.join(map(str,seq))+',')

        for k in keys:

            ret.append( ( list( map( int, k[:-1].split(',') ) ), self.data[ k ] ) )

        return ret

    

    def hasPrefix(self, seq):

        return self.data.has_keys_with_prefix(','.join(map(str,seq))+',')

"""

print()
import pandas as pd



train_df= pd.read_csv('../input/train.csv', index_col="Id", nrows=100)

test_df = pd.read_csv('../input/test.csv', index_col="Id", nrows=100)



train_df= train_df['Sequence'].to_dict()

test_df= test_df['Sequence'].to_dict()

seqs={0: [1 for x in range(0,400)]}



for key in train_df:

    seq=train_df[key]

    seq=[int(x) for x in seq.split(',')]

    seqs[key]=seq



for key in test_df:

    seq=test_df[key]

    seq=[int(x) for x in seq.split(',')]

    seqs[key]=seq



for key in range(2, 5):

    print('ID = '+str(key)+':')

    print(seqs[key])

    print()
import json



trie=prefixTree()

#Caching turned off.

#if not trie.load('trie'):

if True:

    for id in seqs:

        der=seqs[id]

        for derAttempts in range(4):

            seq=der

            firstInTrie=False

            for subseqAttempts in range(4-derAttempts):

                while len(seq)>0 and seq[0]==0:

                    seq=seq[1:]

                signature=findSignature(seq)

                if trie.hasPrefix( signature ):

                    if subseqAttempts==0:

                        firstInTrie=True

                    break

                trie.put( signature, len(seq)*100//len(der) )

                if len(seq)<=3:

                    break

                seq=seq[1:]

            if firstInTrie:

                break

            der=findDerivative(der)

    #trie.save('trie')



print(json.dumps(trie.prefix([2,3,6]),sort_keys=True,indent=2))
from functools import reduce



def findNext(seq, trie):

    while True:

        nonZeroIndex=-1

        for i in range(0,len(seq)):

            if seq[i]!=0:

                nonZeroIndex=i

                break

        if nonZeroIndex<0:

            return 0

        signature=findSignature(seq)

        list=trie.prefix( signature )

        list=filter(lambda x: len(x[0])>len(signature), list)

        item=next(list, None)

        if item!=None:

            best=reduce(lambda a, b: a if a[1]>b[1] else b if b[1]>a[1] else a if len(b[0])<=len(a[0]) else b, list, item)

            nextElement=best[0][len(seq)]

            nextElement*=seq[nonZeroIndex]//signature[nonZeroIndex]

            return nextElement

        if len(seq)<=3:

            break

        seq=seq[1:]

    return None



print(findNext([2,3,6],trie))
def findNextAndDerive(seq, trie):

    nextElement=findNext(seq, trie)

    if nextElement==None:

        der=findDerivative(seq)

        if len(der)<=3:

            return None

        nextElement=findNextAndDerive(der, trie)

        if nextElement==None:

            return None

        return seq[len(seq)-1]+nextElement

    return nextElement



print(findNextAndDerive([1,1,2,3,5,8,13],trie))
trie.put([1,1,2,3,5,8,13],100)

print(findNextAndDerive([1,1,2,3,5,8,13],trie))
total=0

guessed=0

with open('prefix_lookup.csv', 'w+') as output:

    output.write('"Id","Last"\n')

    for id in test_df:

        der=seqs[id]

        nextElement=findNextAndDerive(der, trie)

        output.write(str(id))

        output.write(',')

        total+=1

        if nextElement==None:

            output.write('0')

        else:

            output.write(str(nextElement))

            guessed+=1

        output.write('\n')



print('Total %d' %total)

print('Guessed %d' %guessed)

print('Percent %d' %int(guessed*100//total))