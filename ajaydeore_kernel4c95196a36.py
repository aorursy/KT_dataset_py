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
import pandas as pd
data=pd.read_csv("../input/Social_Network_Ads.csv")
data.head()
data=pd.get_dummies(data)
data.head()
data=data.drop("Gender_Male",axis=1)
data.head()
x=data.iloc[:,0:4].values
x
y=data.iloc[:,-1]
y
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x,y)
purchase=model.predict([[15624575,25,450000,54000]])

if purchase==1:

    print("purchased")

else:

    print("Not Purchased")
data

