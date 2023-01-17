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


train = pd.read_csv("/kaggle/input/2020-ml-p2/train.csv")

X_test = pd.read_csv("/kaggle/input/2020-ml-p2/test.csv")

submit = pd.read_csv("/kaggle/input/2020-ml-p2/test_sampleSubmissin.csv",index_col = 0)
# 학습데이터와 라벨 분리

X_train = train.drop('count', axis=1)

y_train = train['count']



pd.options.display.max_columns = None

train.head(5)


# 모델 학습

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)





# 학습된 모델을 이용한 추론

y_train_pred = regressor.predict(X_train)

y_test_pred = regressor.predict(X_test)





# 대략적인 학습데이터 평가

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_train, y_train_pred))



# 제출파일 저장

submit['count'] = y_test_pred

submit.to_csv("mySubmition.csv")