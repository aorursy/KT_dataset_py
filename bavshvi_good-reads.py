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
books = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines=False)
books.head()
books.info()
books.describe()
top_10 = books.nlargest(n=5, columns=['text_reviews_count'])
top_10['title'] = top_10['title'].str.split('(', expand=True)[0]
top_10
top_10.plot.barh(x = 'title', y='text_reviews_count', rot=0, figsize=(9, 9), color='DarkBlue')