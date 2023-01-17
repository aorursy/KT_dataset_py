# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from collections import Counter

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

#data_types = map(lambda x: {x: type(train_df[x][0])},train_df.keys())

#print (list(data_types))

singularity_train = map(lambda x: {x: len(set(train_df[x]))},train_df.keys())

singularity_test = map(lambda x: {x: len(set(test_df[x]))},test_df.keys())

#print (list(singularity_train))

keys_with_single_train = filter(lambda x: list(x.values())[0]==1, list(singularity_train))

keys_with_single_test = filter(lambda x: list(x.values())[0]==1, list(singularity_test))

print ('these Keys should be ignored:- ',list(keys_with_single_train), list(keys_with_single_test))

y_counter = Counter(train_df['y'])

sorted_y_counter = sorted(y_counter, key=y_counter.get, reverse=True)

print (sorted_y_counter[:10], y_counter[sorted_y_counter[0]])