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
books = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines=False, index_col=0)    # the file path is different on Kaggle



# remove all books with co-authors (names of multiple authors are separated by '/' in the dataset)

books = books[ ~books.authors.str.contains('/') ]



# remove starting and ending spaces in column names

books = books.rename(columns=lambda x: x.strip())