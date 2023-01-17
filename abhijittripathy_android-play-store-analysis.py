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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

data = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
data.head()
data.info()
data.describe()
category_data = data.groupby('Category')
category_data.count()
games_data = data[data['Category']=='GAME']
games_data
games_data.sort_values('Rating',ascending = False)
data5 = games_data[games_data['Rating']==5.0]
data5
# method to slice the size column into a float

def slice_column(patch1):

    new_size = []

    for i in data5[patch1]:

        new_size.append(float(i[:-1]))

    return new_size
data5['Size'] = slice_column('Size')
data5
def plot_game_data(patch1,patch2,data,kind):

    sns.jointplot(x = patch1,y = patch2,data = data,kind = kind)

#     plt.xlabel(patch1,fontsize=15)

#     plt.ylabel(patch2,fontsize=15)

#     plt.legend(fontsize=15)

#     plt.title('{} VS {} Plot'.format(patch1,patch2),fontsize=15)

    plt.show()
# Reviews column is of object type, lets make that float

def make_float():

    new_review = []

    for i in data5.Reviews:

        new_review.append(float(i))

    return new_review
data5['Reviews'] = make_float()
# plotting the size column along with reviews

plot_game_data('Reviews','Size',data5,'scatter')
data5['Installs'] = slice_column('Installs')
# plotting a graph between size and installs

plot_game_data('Size','Installs',data5,'scatter')