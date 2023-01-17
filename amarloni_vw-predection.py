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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
vw = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/vw.csv', sep = ',')

vw.info()
vw.isnull().sum()
vw.head()

vw1 = vw.copy()
vw.isna().sum()
from sklearn.preprocessing import StandardScaler, LabelEncoder

le = LabelEncoder()

sc = StandardScaler()



vw.model = le.fit_transform(vw['model'])

vw.transmission = le.fit_transform(vw['transmission'])

vw.fuelType = le.fit_transform(vw['fuelType'])
plt.figure(figsize=(10,7))

corr = vw.corr()

sns.heatmap(corr, annot=True, linewidth =0.5)
#sns.pairplot(vw1)
columns = vw.columns.unique()

for column in columns:

    plt.figure(figsize=(10,7))

    sns.scatterplot(x='price', y= column, hue= 'year', data = vw, palette="deep")

    plt.show()
vw = pd.DataFrame(vw)

vw.head()

X = vw.drop(['price'],1)

y = vw['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 43)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, BayesianRidge, ridge_regression

from sklearn import metrics

from sklearn.metrics import r2_score, mean_squared_error



lr = LogisticRegression(max_iter=100, solver='liblinear')

rd = Ridge()

la = Lasso()

byrd = BayesianRidge()



models = [lr, rd, la, byrd]

for model in models:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mod = model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2_score = metrics.r2_score(y_test, y_pred)

    RMSE = metrics.mean_squared_error(y_test,y_pred)

    print('\n', model, 'R2_score:', r2_score,'\n', 'RMSE:', '\n', RMSE, '\n')