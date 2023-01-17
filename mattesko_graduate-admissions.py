# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import Normalizer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Admission_Predict.csv')



# Fix the column names (Admit had a space after it)

data.columns = ['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',

       'LOR ', 'CGPA', 'Research', 'Chance of Admit']
# Number of top correlated variables for heatmap

k = len(data.columns)

corr_matrix = data.corr()

cols = corr_matrix.nlargest(k, 'Chance of Admit')['Chance of Admit'].index

cm = np.corrcoef(data[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
data.drop(columns=['Serial No.'], inplace=True)
X_train, X_test, y_train, y_test = train_test_split(

    data.drop(columns=['Chance of Admit']).values, 

    data['Chance of Admit'].values, 

    test_size=0.2)
pipeline_linear_reg = Pipeline([

    ('norm', Normalizer()),

    ('estim', LinearRegression())

])
pipeline_linear_reg.fit(X_train, y_train);
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pipeline_linear_reg.predict(X_test)))