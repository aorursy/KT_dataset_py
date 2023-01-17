print("Hi guys")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
print("Hello team!")
df1 = pd.read_csv('../input/netflix-prize-data/combined_data_1.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
df1.shape
df2 = pd.read_csv('../input/netflix-prize-data/combined_data_2.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
df3 = pd.read_csv('../input/netflix-prize-data/combined_data_3.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
df4 = pd.read_csv('../input/netflix-prize-data/combined_data_4.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
df = pd.concat([df1, df2, df3, df4], axis = 0)
df.shape
# Extract Movie ID:
nan = pd.DataFrame(pd.isnull(df['Rating']))
movie_id = df[nan['Rating'] == True]['Cust_Id']
movie_id = movie_id[:1001,]
Movie_ID = []
for i in range(1,1000):
    Movie_ID.append(movie_id.index[i]*[i])
Movie_ID[0]