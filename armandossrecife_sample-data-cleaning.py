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
movies_df = pd.read_csv('/kaggle/input/imdb-5000-movie-dataset/movie_metadata.csv')
movies_df
columns_my_dataset = movies_df.columns

columns_my_dataset
movies_df.info()
movies_df.describe()
movies_df.isnull()
movies_df.loc[4]
movies_df.isnull().sum()
cleaned_movies_df = movies_df.dropna(how='any')

cleaned_movies_df
movies_df.duplicated()
duplicated_rows_movies_df = movies_df[movies_df.duplicated()]
movies_df.query('director_name=="Albert Hughes"')
import seaborn as sns
sns.boxplot(x=movies_df.facenumber_in_poster, color='green')