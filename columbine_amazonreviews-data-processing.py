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
dataset = pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')
dataset.head(5)
dataset.iloc[0]['Text']
from nltk.tokenize import sent_tokenize, word_tokenize
sent_list_exp = sent_tokenize(dataset.iloc[0]['Text'])
word_tokenize(text = sent_list_exp[1])
AmazonDataset = dataset[['Text', 'Score']]
AmazonDatasetText = dataset['Text']



AmazonDatsetTarget = dataset['Score']
from sklearn.model_selection import train_test_split
train, test = train_test_split(AmazonDataset, test_size=0.4, stratify=AmazonDatsetTarget) 



dev, test = train_test_split(test, test_size=0.5)
train.index = np.arange(0, len(train))



dev.index = np.arange(0, len(dev))



test.index = np.arange(0, len(test))
train.to_csv('/kaggle/working/train.csv')



dev.to_csv('/kaggle/working/dev.csv')



test.to_csv('/kaggle/working/test.csv')