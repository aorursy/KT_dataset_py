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
test = pd.read_csv('/kaggle/input/student-shopee-code-league-sentiment-analysis/test.csv')

train = pd.read_csv('/kaggle/input/student-shopee-code-league-sentiment-analysis/train.csv')

sub = pd.read_csv("/kaggle/input/student-shopee-code-league-sentiment-analysis/sampleSubmission.csv")
train
(train['rating'].value_counts()/len(train)).sort_index()
test.head()
test.tail()
proportions = (train['rating'].value_counts()/len(train)).sort_index()
ans_1 = [1]* int(round(proportions[1]*len(test)))

ans_2 = [2]* int(round(proportions[2]*len(test)))

ans_3 = [3]* int(round(proportions[3]*len(test)))

ans_4 = [4]* int(round(proportions[4]*len(test)))

ans_5 = [5]* int(round(proportions[5]*len(test)))
pred = ans_1+ans_2+ans_3+ans_4+ans_5
test['rating'] = pred
test
test[['review_id', 'rating']].to_csv("submission.csv", index=False)