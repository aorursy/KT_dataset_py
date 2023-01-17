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
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
Advertising  = pd.DataFrame(pd.read_csv("../input/advertising-dataset/advertising.csv"))
Advertising .head()
Advertising.describe()
sns.pairplot(Advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales')
model = lr()
x = Advertising[["TV","Radio","Newspaper"]]
y = Advertising["Sales"]
model.fit(x,y)
print(model.intercept_)
print(list(zip(x, model.coef_)))
from sklearn import model_selection

xtrain,xtest,ytrain,ytest=model_selection.train_test_split(x,y,random_state=1)
model1 = lr()
model1.fit(xtrain,ytrain)
print(model1.intercept_,",", model1.coef_)
predict = model1.predict(xtest)

from sklearn.metrics import mean_squared_error as mse
from math import sqrt

print(sqrt(mse(ytest,predict)))
model.score(x,y)
