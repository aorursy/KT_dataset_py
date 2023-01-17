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
import pandas as pd
twosigma_train=pd.read_json('/kaggle/input/two-sigma-connect-rental-listing-inquiries/train.json.zip')

print('Train Data : ',twosigma_train.shape)
twosigma_test=pd.read_json('/kaggle/input/two-sigma-connect-rental-listing-inquiries/test.json.zip')

print('Test Data : ',twosigma_test.shape)
print(twosigma_train.columns.to_list())

print(twosigma_train.interest_level.value_counts())
twosigma_train.describe()
twosigma_train.head()
import matplotlib.pyplot as plt

plt.style.use('ggplot')
prices=twosigma_train.groupby('interest_level',as_index=False)['price'].median()
#barplot

fig=plt.figure(figsize=(8,6))

plt.bar(prices.interest_level,prices.price,color='blue',width=0.5,alpha=0.8)

#set titles

plt.xlabel('Interest level')

plt.ylabel('Median price')

plt.title('Median listing price across interest level')

plt.show()
plt.figure(figsize=(8,6))

plt.scatter(x=twosigma_train['price'],y=twosigma_train['interest_level'],color='green',alpha=0.5)

plt.xlabel('price')

plt.ylabel('interest_level')

plt.title('comparison b/w price and interest level')

plt.show()