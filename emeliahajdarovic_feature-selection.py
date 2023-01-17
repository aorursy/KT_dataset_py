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
df=pd.read_csv("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")

df
df.info()
df.columns
chi_df=df[['PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','default.payment.next.month']]

chi_df
#libraries

import numpy as numpy

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import LabelEncoder
pay=['PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder



for i in pay:

    label_encoder = LabelEncoder()

    chi_df[i] = label_encoder.fit_transform(chi_df[i])

    

chi_df.head()
from sklearn.feature_selection import chi2

X = chi_df.drop('default.payment.next.month',axis=1) #df without default col

y = chi_df['default.payment.next.month'] #default col
#get chi scores

chi_scores = chi2(X,y)
chi_scores
p_values = pd.Series(chi_scores[1],index = X.columns)

p_values.sort_values(ascending = False , inplace = True)

p_values.plot.bar()