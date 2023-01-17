# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["wc", "-l", "../input/train.csv"]).decode("utf8"))
print(check_output(["wc", "-l", "../input/test.csv"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
trainset= pd.read_csv('../input/train.csv', index_col= 'Id')
trainset.head()
fp = open('../input/train.csv', 'r')
header = fp.next().strip().split(',')
df = pd.DataFrame()
for line in fp:
    rec = line.strip().split(',')
    id_val = rec[0]
    val = rec[1]
    val = map(int, val.split(','))[::-1]
    df = pd.concat([df, pd.DataFrame(zip(id_val, val))])
    
import csv
fp = csv.reader(open('../input/train.csv', 'r'))
#header = fp.next()
#print fp.next()
#print(header)

#fp.readline()
count= 1
for i in fp:
    print(i)
    count += 1
    if count >= 5: 
        break
type(fp)
trainset.shape
testset= pd.read_csv('../input/test.csv', index_col= 'Id')
testset.head()
testset.shape
seq= trainset.Sequence.ix[3]
seq
trainset['seqlen']= trainset.Sequence.apply(lambda x: len(str.split(x, ',')))
trainset.head()
trainset.seqlen.value_counts()
trainset.seqlen.nunique()
trainset.seqlen.describe()
trainset= trainset[trainset.seqlen > 1]
trainset.seqlen.describe()
trainset['lastval']= trainset.Sequence.apply(lambda x: str.split(x, ',')[-1] )
trainset['firstval']= trainset.Sequence.apply(lambda x: str.split(x, ',')[0] )
trainset.head()
trainset['uniqvals']= trainset.Sequence.apply(lambda x: len(set(str.split(x, ','))) )
trainset.head()

