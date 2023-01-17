import pandas as pd



books = pd.read_csv('/kaggle/input/market-basket-analysis-dataset/bookstore_transactions.csv')

print(books.head(2))
transactions = books['Transaction'].apply(lambda t: t.split(','))

transactions = list(transactions)

transactions
history = transactions.count(['History', 'Bookmark'])

biography = transactions.count(['Biography', 'Bookmark'])

fiction = transactions.count(['Fiction', 'Bookmark'])



print('history:', history)

print('biography:', biography)

print('fiction:', fiction)
from itertools import permutations



flattened = [item for transaction in transactions for item in transaction]

items = list(set(flattened))
rules = list(permutations(items,2))

print(rules)
print(len(rules))
from mlxtend.preprocessing import TransactionEncoder

encoder = TransactionEncoder().fit(transactions)

onehot = encoder.transform(transactions)
onehot = pd.DataFrame(onehot, columns=encoder.columns_)

print(onehot)
print(onehot.mean())
import numpy as np



onehot['Fiction+Poetry'] = np.logical_and(onehot['Fiction'],onehot['Poetry'])



print(onehot.mean())
supportBF = np.logical_and(onehot['Biography'], onehot['Fiction']).mean()

supportBF
supportBP = np.logical_and(onehot['Biography'], onehot['Poetry']).mean()

supportBP
supportBH = np.logical_and(onehot['Biography'], onehot['History']).mean()

supportBH
supportPH = np.logical_and(onehot['Poetry'], onehot['History']).mean()

supportPH
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session