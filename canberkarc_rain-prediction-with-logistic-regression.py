# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # splittind data
from sklearn.linear_model import LogisticRegression

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv")
df.info()
df.RainTomorrow = df.RainTomorrow.replace(to_replace = ['Yes','No'],value = ['1','0'])
df.columns
df.RISK_MM = [int(each*10) for each in df.RISK_MM]

x = df.RainTomorrow
y = df.RISK_MM

#Splitting test and train data
x_train, x_test, y_train, y_test = train_test_split(x.T,y,test_size = 0.2, random_state = 42)

#Converting strings to numerical values
x_train = x_train.replace(to_replace = ['Yes','No'],value = ['1','0'])
x_test = x_test.replace(to_replace = ['Yes','No'],value = ['1','0'])

#We multiply values by 100 to not lose values while converting to int and get more accuracy.
x_train = [int(each*100) for each in x_train]
x_test = [int(each*100) for each in x_test]
y_train = [int(each*100) for each in y_train]
y_test = [int(each*100) for each in y_test]

#Sklearn needs 2d array so we convert list to array to use reshape method and get 2d arrays
x_t_arr = np.array(x_train).reshape(-1,1)
y_t_arr = np.array(y_train).reshape(-1,1)
x_test_arr = np.array(x_test).reshape(-1,1)
y_test_arr = np.array(y_test).reshape(-1,1)



lr = LogisticRegression(solver='lbfgs', max_iter=3000)
lr.fit(x_t_arr,y_t_arr)
print("Score : ",lr.score(x_test_arr,y_test_arr))
#Score is nearly 0.639