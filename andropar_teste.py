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
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import linear_model
%matplotlib inline

df = pd.read_csv('../input/GlobalTemperatures.csv')
df.info()
df.head()
df['dt'] = pd.to_datetime(df['dt'])
df['year'] = df['dt'].dt.year
df_year_AveTemp = df.loc[:,['year','LandAverageTemperature']]
df_year_AveTemp = df_year_AveTemp.dropna()
df_year_AveTemp.head()
df_year_AveTemp.plot(kind='scatter', x='year', y='LandAverageTemperature', figsize=(16,8))
temp = df_year_AveTemp.groupby('year')
df_year_AveTemp = temp.mean()
df_year_AveTemp['year'] = df_year_AveTemp['LandAverageTemperature'].index
import statsmodels.formula.api as smf
linearRegression = smf.ols(formula='LandAverageTemperature ~ year', data = df_year_AveTemp).fit()
linearRegression.params
minMaxYear = pd.DataFrame({'year': [df_year_AveTemp.year.min(),
                                                 df_year_AveTemp.year.max()]})
predictions = linearRegression.predict(x_new)
df_year_AveTemp.plot(kind='scatter', x='year', y='LandAverageTemperature', figsize=(16,8))
plt.plot(minMaxYear, predictions, c='red', linewidth=2)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
train, test = train_test_split(df_year_AveTemp, test_size = 0.2)
train_x = np.array(train['year']).reshape(-1, 1)
train_y = np.array(train['LandAverageTemperature'], dtype="|S6")
test_x = np.array(test['year']).reshape(-1, 1)
randomForest = RandomForestClassifier(n_estimators = 1000)
randomForest.fit(train_x, train_y)
predictions = randomForest.predict(test_x)
predictions
