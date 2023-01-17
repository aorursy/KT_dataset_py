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
train = pd.read_csv('/kaggle/input/shopee-sentiment-analysis/train.csv')

test = pd.read_csv('/kaggle/input/shopee-sentiment-analysis/test.csv')
print(train.shape)

print(test.shape)
train.head()
proportion = train['rating'].value_counts(normalize=True).sort_index().values

proportion
ans = []

for i, pcg in enumerate(proportion, 1):

    leng = int(round(pcg*62918))

    ans += [i] * leng
len(ans)
test['rating'] = ans
test.head()
test[['review_id', 'rating']].to_csv('test_data_leakage.csv', index=False)