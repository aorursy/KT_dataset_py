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
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
train.head()
test.head()
import re

# Remove urls

test['text'] = test['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
submission.head()
avg_len = np.mean(train['text'].apply(lambda x:len(str(x).split(' '))))
submission["selected_text"] = test["text"].apply(lambda x: " ".join(x.split()[-int(avg_len*2):]))
submission.sample(5)
submission.to_csv('submission.csv', index=False)