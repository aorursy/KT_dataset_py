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
from sklearn.metrics import r2_score

import numpy as np
y = np.random.randint(90,100,200)

y[100] = 250 

# the prediction has only one value different from the real y

y_predict = pd.Series(y).apply(lambda x: x-150 if x > 150 else x)
np.corrcoef(y, y_predict)
# see what happend when we calculated the r2 score

# pretty low, right? that's the weakness of r2 score

# hope there wouldn't be too many outlier in our test dataset

print(r2_score(y, y_predict))
np.random.seed(0)

y = np.array([100]*4000) + np.random.randint(0,20,4000)

y_preidct = y + np.random.randint(0,7,4000)
print(r2_score(y, y_preidct))

print(r2_score(y[:800], y_preidct[:800]))

print(r2_score(y[800:], y_preidct[800:]))
# let change one data point to see the impact to our whole dataset and sub datast

# I treat the sub dataset as out LB data, the impact is huge, right?

y_preidct[600] = 150

print(r2_score(y, y_preidct))

print(r2_score(y[:800], y_preidct[:800]))

print(r2_score(y[800:], y_preidct[800:]))
# suppose our overfitting model predict the outlier in [:800] right and also create two fake outlier in the rest dataset

# the wrong outlier we predict harm a lot to our private LB data

y_preidct[600] = 150

y_preidct[1300] = 150

y_preidct[2400] = 150

y[600] = 150

print(r2_score(y, y_preidct))

print(r2_score(y[:800], y_preidct[:800]))

print(r2_score(y[800:], y_preidct[800:]))