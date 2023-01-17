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


# change to your actual file name

in_filename = '/kaggle/input/amazon-fine-food-reviews/Reviews.csv'



# read only the Text column from the csv to a data frame

data = pd.read_csv(in_filename, usecols=['Text'])









keywords = ['food', 'bought', 'error', 'good', 'flavor', 'fresh', 'tasty', 'healthy', 'great', ]

# add columns with value 0 for each keyword

for i, k in enumerate(keywords):

    data.insert(i+1, k, 0)

    

#print(data.columns)

#print(len(data['Text']))



# for each string

for i, s in enumerate(data['Text'][:10] ):

    # for each keyword

    for k in keywords:

        # data for row i column k is the count of k in s

        data.loc[i, k] = s.count(k)

print(data.loc[:10])





# write to file

open("/kaggle/input/amazon-fine-food-reviews/Reviews.csv", "w").write("contents")

data.to_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv', encoding='utf-8', index=False)