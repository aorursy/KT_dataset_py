# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages bto load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import imagehash
pd.read_csv('../input/tweets.csv').columns
data = pd.read_csv('../input/tweets.csv')
data.count()
for col in data.columns:

    print("{0} unique vals in {1}".format(data[col].drop_duplicates().count(), col))
for name in data['name'].drop_duplicates():

    s = sum(data.tweets.str.contains(name))

    if (s > 0):

        print("{0} has sum {1}".format(name,s))
import networkx as nx
for name in data['username'].drop_duplicates():

    s = sum(data.tweets.str.contains("@" + name))

    if (s > 0):

        print("{0} has sum {1}".format(name,s))