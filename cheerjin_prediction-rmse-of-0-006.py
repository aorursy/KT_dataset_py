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
import numpy as np

from sklearn import preprocessing

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df.head()
continuous_vars = ['GRE Score', 'TOEFL Score', 'CGPA']

categorical_vars = ['University Rating', 'SOP', 'LOR ', 'Research']

target = 'Chance of Admit '
continuous_df = df[continuous_vars]

continuous_df = preprocessing.StandardScaler().fit_transform(continuous_df)

continuous_df
categorical_df = df[categorical_vars]

enc = preprocessing.OneHotEncoder(drop='first').fit(categorical_df)

enc_categorical_df = enc.transform(categorical_df).toarray()

enc_categorical_df
print(continuous_df.shape, enc_categorical_df.shape)
processed_data = np.concatenate((continuous_df, enc_categorical_df), axis=1)

processed_data.shape
y = df[target]
X_train = processed_data[0:400]

X_test = processed_data[400:500]

y_train = y.iloc[0:400]

y_test = y.iloc[400:500]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
reg = MLPRegressor(max_iter=1000).fit(X_train, y_train)
y_pred = reg.predict(X_test)
mean_squared_error(y_test, y_pred)