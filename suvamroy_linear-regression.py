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

apndcts = pd.read_csv("../input/apndcts/apndcts.csv")

apndcts

apndcts.info()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

y= apndcts.pop('class')

x= apndcts

x_train,x_test,y_train ,y_test=train_test_split(x,y)

model=LogisticRegression()

model.fit(x_train,y_train)

acu=model.score(x_test,y_test)

print(acu)