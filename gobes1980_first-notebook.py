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
data = pd.read_csv('../input/movie_metadata.csv',parse_dates=True)
data.info()
data.plot(kind='scatter',x='imdb_score',y='gross',figsize=[10,10])
data.plot(kind='scatter',y='imdb_score',x='budget',figsize=[10,10],logx=True)
import matplotlib.pyplot as plt
finance = data[['movie_title','budget','gross','imdb_score','title_year']]
finance.describe()
%matplotlib inline

finance[['budget','gross']].plot(kind='box',logy=True,figsize=[10,10])

data.plot(kind='scatter',x='budget',y='gross',logx=True,figsize=(10,10))