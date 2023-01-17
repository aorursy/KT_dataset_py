# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
medical_rec = pd.read_csv("/kaggle/input/insurance/insurance.csv")
medical_rec.head(5)
fig = plt.figure(figsize=(14,6))

sns.violinplot(x="sex", y="charges", data=medical_rec, hue="smoker")
fig = plt.figure(figsize=(14,6))

sns.scatterplot(x="bmi", y="charges", data=medical_rec, hue="smoker")
categorical_columns = ['sex','children', 'smoker', 'region']

medical_rec_encoded = pd.get_dummies(data = medical_rec, prefix = None, prefix_sep='_',

               columns = categorical_columns,

               drop_first =True,

              dtype='int8')

medical_rec_encoded.head(5)
medical_rec_encoded['charges'] = np.log(medical_rec_encoded['charges'])
medical_rec_encoded.head(5)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(medical_rec_encoded.drop("charges", axis=1), medical_rec_encoded["charges"])
Lin = LinearRegression()
Lin.fit(X_train, y_train)
prediction = Lin.predict(X_test)
evaluation_metrics = pd.DataFrame({"prediction":prediction, "actual":y_test}).reset_index(drop=True)

evaluation_metrics.head(5)
import seaborn as sns

import matplotlib.pyplot as plt
sns.lineplot(x=evaluation_metrics.index, y=evaluation_metrics["prediction"]-evaluation_metrics['actual'])
from sklearn import metrics

MAE = metrics.mean_absolute_error(y_test, prediction)

MSE = metrics.mean_squared_error(y_test, prediction)

RMSE = np.sqrt(metrics.mean_squared_error(y_test, prediction))

print("MAE: {}".format(MAE))

print("MSE: {}".format(MSE))

print("RMSE: {}".format(RMSE))