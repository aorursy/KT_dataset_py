# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Data Preprocessing

dataset = pd.read_csv('/kaggle/input/suggestions/Market_Basket_Optimisation.csv', header = None)

transactions = []

for i in range(0, 7501):

  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
!pip install apyori
# Training the Eclat model on the dataset

from apyori import apriori

rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
## Displaying the first results coming directly from the output of the apriori function

results = list(rules)

results
## Putting the results well organised into a Pandas DataFrame

def inspect(results):

    lhs         = [tuple(result[2][0][0])[0] for result in results]

    rhs         = [tuple(result[2][0][1])[0] for result in results]

    supports    = [result[1] for result in results]

    return list(zip(lhs, rhs, supports))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])
## Displaying the results sorted by descending supports

resultsinDataFrame.nlargest(n = 10, columns = 'Support')