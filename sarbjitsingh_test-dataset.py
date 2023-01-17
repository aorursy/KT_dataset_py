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
data = pd.read_csv('../input/kc_house_data.csv')

data.head()
data.info()
y = data['price']

y.head()
features = ['lat', 'waterfront', 'grade', 'long', 'view', 'bathrooms']

x = data[features]

x.head()
import seaborn as sbn



sbn.pairplot(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3)
from sklearn.linear_model import LinearRegression



reg = LinearRegression()

reg.fit(x_train, y_train)





accuracy  = reg.score(x_test, y_test)

print(accuracy)