# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install ernie
from ernie import SentenceClassifier, Models

import pandas as pd
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv",usecols=['text','target'])

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sub_sample = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")



print (train.shape, test.shape, sub_sample.shape)

train = train.rename(columns={'text':0,'target':1})

train.head(5)
import seaborn as sns

sns.countplot(y=train[1])
classifier = SentenceClassifier(model_name=Models.BertBaseUncased, max_length=128, labels_no=2)
classifier.load_dataset(dataframe=train ,validation_split=0.2)
classifier.fine_tune(epochs=5, learning_rate=2e-5, training_batch_size=32, validation_batch_size=64)
from ernie import SplitStrategies, AggregationStrategies



text = "ablaze,London,Birmingham Wholesale Market is ablaze BBC News -Fire breaks out at Birmingham's Wholesale Market http://t.co/irWqCEZWEU"



probabilities = classifier.predict_one(text,

                                   aggregation_strategy=AggregationStrategies.Mean)
probabilities