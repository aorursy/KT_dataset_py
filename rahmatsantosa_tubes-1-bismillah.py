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
from scipy.io import arff



data,meta = arff.loadarff('/kaggle/input/weather.arff')

data



df = pd.DataFrame(data)

df
#preprocess categorize and one hot encode attribute



from sklearn.preprocessing import OneHotEncoder



#categorize column play

for col in ['play']:

    df[col] = df[col].astype('category')



df['play'] = df['play'].cat.codes



#categorize column windy

for col in ['windy']:

    df[col] = df[col].astype('category')

    

df['windy'] = df['windy'].cat.codes



pre_processed = pd.get_dummies(df,prefix=['outlook'])



pre_processed
label = pre_processed['play']

pd.DataFrame(label)