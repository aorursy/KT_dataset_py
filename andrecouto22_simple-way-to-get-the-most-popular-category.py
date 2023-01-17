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
stats_dir = '../input/google-play-store-apps/googleplaystore.csv'

reviews_dir = '../input/google-play-store-apps/googleplaystore_user_reviews.csv' 



stats_data = pd.read_csv(stats_dir)

reviews_data = pd.read_csv(reviews_dir)
#Get list of all categories

Cat = list(stats_data["Category"].unique())



#sum amount of apps per

dict_cat = {}

for i in Cat:

        num = stats_data["Category"] == i

        dict_cat[i] = num.sum()

dict_cat = sorted(dict_cat.items(), key=lambda x: x[1],reverse=True)
print(f"Most popular category is {dict_cat[0][0]} with {dict_cat[0][1]} apps" )