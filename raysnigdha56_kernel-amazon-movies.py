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
!pip list |grep pandas
%pip install -U pip 

%pip install pandas==0.24.0
import pandas as pd

import numpy as np
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
amazon = pd.read_csv('/kaggle/input/data-sets-for-practice/Amazon - Movies and TV Ratings.csv')

amazon.head()
amazon.describe()
amazon.describe().T["count"].sort_values(ascending = False)[:10]
amazond = amazon.drop('user_id', axis = 1)

amazond.head()
amazond.sum().sort_values(ascending = False).to_frame()[:20]
!pip install scikit-surprise
from surprise import Reader

from surprise import accuracy

from surprise.model_selection import train_test_split
help(amazon.melt())
amazon.columns
melt_amazon = amazon.melt(id_vars = amazon.columns[0], value_vars = amazon.columns[1:], var_name="movie name", value_name="ratings")
melt_amazon
from surprise import Dataset
reader = Reader(rating_scale=(-1,10))

data = Dataset.load_from_df(melt_amazon.fillna(0), reader = reader)
trainset, testset = train_test_split(data, test_size = 0.25)
from surprise import SVD
algo = SVD()
algo.fit(trainset)
prediction = algo.test(testset)

accuracy.rmse(prediction)
user_id = 'A3R5OBKS7OM2IR'

movie_id = 'Movie1'

rating = 5.0

algo.predict(user_id, movie_id, r_ui=rating, verbose = True)

# here it says the accuracy (stimated value as per the actual value which is 0.060, which is not good

#though the rmse value is also not good)
from surprise.model_selection import cross_validate
cross_validate(algo, data, measures=['RMSE','MAE'], cv = 3, verbose = False)
def repeat(algo_type, frame, min_, max_):

    reader = Reader(rating_scale=(min_,max_))

    data = Dataset.load_from_df(frame, reader=reader)

    algo = algo_type

    print(cross_validate(algo, data, measures=['RMSE','MAE'], cv = 3, verbose = True))

    user_id = 'A3R5OBKS7OM2IR'

    movie_id = 'Movie1'

    rating = 5.0

    algo.predict(user_id, movie_id, r_ui=rating, verbose = True)

amazon = amazon.iloc[:1212, :50]

melt_amazon = amazon.melt(id_vars = amazon.columns[0], value_vars = amazon.columns[1:], var_name="movie name", value_name="ratings")

repeat(SVD(), melt_amazon.fillna(0), -1, 10)

repeat(SVD(), melt_amazon.fillna(melt_amazon.mean()), -1, 10)

repeat(SVD(), melt_amazon.fillna(melt_amazon.median()), -1, 10)