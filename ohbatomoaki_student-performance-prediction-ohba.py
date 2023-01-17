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
df_train = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/test.csv', index_col=0)
import seaborn as sns

from matplotlib import pyplot



sns.set_style("darkgrid")

pyplot.figure(figsize=(31, 31))

sns.heatmap(df_train.corr(), square=True, annot=True)
df_train = df_train[['age','Medu','Fedu','studytime','failures','higher','G3']]

df_test = df_test[['age','Medu','Fedu','studytime','failures','higher']]
df_train = pd.get_dummies(df_train, drop_first=True)

df_test = pd.get_dummies(df_test, drop_first=True)
from sklearn.svm import SVR



X_train = df_train.drop('G3', axis=1).values

y_train = df_train['G3'].values



model = SVR(kernel='rbf',gamma='auto')

model.fit(X_train, y_train)
X_test = df_test.values

predict = model.predict(X_test)
submit = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/sampleSubmission.csv')

submit['G3'] = predict

submit.to_csv('submission.csv', index=False)