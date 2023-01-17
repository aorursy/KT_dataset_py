# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for Israel, _, Medical_cost in os.walk('/kaggle/input'):
    for Medical_cost in Medical_cost:
        print(os.path.join(Israel, Medical_cost))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
data = pd.read_csv('../input/insurance/insurance.csv')
data
data['sex'] = data['sex'].map({'female':1, 'male':0})
data['smoker'] = data['smoker'].map({'yes':1, 'no':0})
data
new_data = data.copy()
new_data.describe()
y = new_data['charges']
x = new_data['sex']
x_matrix = x.values.reshape(-1,1)
x_matrix.shape

reg = LinearRegression()
reg.fit(x_matrix, y)
reg.score(x_matrix,y)
reg.coef_
reg.intercept_
reg.predict(x_matrix)
data_new = pd.DataFrame(x_matrix, columns=['sex'])
data_new
reg.predict(data_new)
data_new['Predicted'] = reg.predict(data_new)
data_new
