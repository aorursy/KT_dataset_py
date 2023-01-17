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
playStore = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")

userReviews = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv")
playStore.head(10)
playStore.info()
playStore[['Rating']].sort_values('Rating', ascending = False)
playStore.describe(include = 'all')
playStore.dropna(subset=['Reviews'], inplace=True)
playStore.info()
pd.to_numeric(playStore['Reviews'],errors='coerce').max()
playStore['Reviews'] = pd.to_numeric(playStore['Reviews'], errors='coerce')
playStore.sort_values(by = 'Reviews', ascending = False).iloc[0]
appWithMostReviews = playStore.sort_values(by = 'Reviews', ascending = False).iloc[0]['App']

appWithMostReviews
print(appWithMostReviews + " is the app with most reviews!")