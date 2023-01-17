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
df = pd.read_csv('../input/data-exploration/avgratings.csv')

df.count()
from surprise import Reader, Dataset, SVD
reader = Reader()

data = Dataset.load_from_df(df[['user_id', 'venue_id', 'rating']], reader)

svd = SVD()

trainset = data.build_full_trainset()

svd.fit(trainset)
from surprise import dump

file_name = 'movie_svd_model'

dump.dump(file_name, algo=svd)