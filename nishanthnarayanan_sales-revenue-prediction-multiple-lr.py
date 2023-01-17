# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = (12, 8)
ad = pd.read_csv("/kaggle/input/sales-revenue-dataset/Advertising.csv")
ad.head()
ad.info()
sns.pairplot(ad, x_vars=['TV','radio','newspaper'], y_vars='sales', height = 7, aspect = 0.7)
plt.show()
from sklearn.linear_model import LinearRegression

x = ad[['TV','radio','newspaper']]
y= ad.sales

lm1 = LinearRegression()
lm1.fit(x,y)

print(lm1.intercept_)
print(lm1.coef_)
list(zip(['TV','radio','newspaper'], lm1.coef_))
sns.heatmap(ad.corr(), annot = True)
plt.show()
from sklearn.metrics import r2_score

lm2 = LinearRegression().fit(x[['TV','radio']], y)
lm2_pred = lm2.predict(x[['TV','radio']])

print("R2: ", r2_score(y, lm2_pred))
lm3 = LinearRegression().fit(x, y)
lm3_pred = lm3.predict(x)

print("R2: ", r2_score(y, lm3_pred))
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)

lm4 = LinearRegression().fit(x_train,y_train)

lm4_preds = lm4.predict(x_test)

print("RMSE: ", np.sqrt(mean_squared_error(y_test, lm4_preds)))

print("R2: ", r2_score(y_test, lm4_preds))
x_train, x_test, y_train, y_test = train_test_split(x[['TV', 'radio']], y, random_state = 1)

lm5 = LinearRegression().fit(x_train,y_train)

lm5_preds = lm5.predict(x_test)

print("RMSE: ", np.sqrt(mean_squared_error(y_test, lm5_preds)))

print("R2: ", r2_score(y_test, lm5_preds))
from yellowbrick.regressor import PredictionError, ResidualsPlot

visualizer = PredictionError(lm5).fit(x_train, y_train)
visualizer.score(x_test, y_test)
visualizer.poof()
plt.show()
ad['interaction'] = ad['TV'] * ad['radio']

x = ad[['TV','radio','interaction']]
y = ad.sales

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)

lm6 = LinearRegression().fit(x_train,y_train)

lm6_preds = lm6.predict(x_test)

print("RMSE: ", np.sqrt(mean_squared_error(y_test, lm6_preds)))

print("R2: ", r2_score(y_test, lm6_preds))
visualizer = PredictionError(lm6).fit(x_train, y_train)
visualizer.score(x_test, y_test)
visualizer.poof()
plt.show()