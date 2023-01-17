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
from surprise import SVD

from surprise import Dataset

from surprise import evaluate, print_perf





# Load the movielens-100k dataset (download it if needed),

# and split it into 3 folds for cross-validation.

df_train = pd.read_csv('../input/train.csv')
import surprise

algo = surprise.SVD()

reader = surprise.Reader(rating_scale=(0,1))

data = surprise.Dataset.load_from_df(df_train[['msno', 'song_id', 'target']].dropna(), reader)

trainset = data.build_full_trainset()

algo.train(trainset)

print("Done Training")



df_test = pd.read_csv('../input/test.csv')

submit = []

for index, row in df_test.iterrows():

    est = algo.predict(row['msno'], row['song_id']).est

    submit.append((row['id'], est))

submit = pd.DataFrame(submit, columns=['id', 'target'])

submit.to_csv('submission.csv', index=False)

print("Created submission.csv")