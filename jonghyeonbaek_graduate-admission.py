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

data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
data
data.info()
data.describe()
data.isnull().sum()
#상위 25% 기준으로 데이터를 분석한다.

large = data[data['Chance of Admit '] >= 0.75]['GRE Score']
smaller= data[data['Chance of Admit '] < 0.75]['GRE Score']
print(large.describe())
print(smaller.describe())
plt.plot(np.arange(0,len(large)),large)
plt.show()
plt.plot(np.arange(0,len(smaller)),smaller)
plt.show()
large = data[data['Chance of Admit '] >= 0.75]['TOEFL Score']
smaller= data[data['Chance of Admit '] < 0.75]['TOEFL Score']
print(large.describe())
print(smaller.describe())
plt.plot(np.arange(0,len(large)),large)
plt.show()
plt.plot(np.arange(0,len(smaller)),smaller)
plt.show()
large = data[data['Chance of Admit '] >= 0.75]['University Rating']
smaller= data[data['Chance of Admit '] < 0.75]['University Rating']
print(large.describe())
print(smaller.describe())
plt.plot(np.arange(0,len(large)),large)
plt.show()
plt.plot(np.arange(0,len(smaller)),smaller)
plt.show()
large = data[data['Chance of Admit '] >= 0.75]['SOP']
smaller= data[data['Chance of Admit '] < 0.75]['SOP']
print(large.describe())
print(smaller.describe())
plt.plot(np.arange(0,len(large)),large)
plt.show()
plt.plot(np.arange(0,len(smaller)),smaller)
plt.show()
large = data[data['Chance of Admit '] >= 0.75]['LOR ']
smaller= data[data['Chance of Admit '] < 0.75]['LOR ']
print(large.describe())
print(smaller.describe())
plt.plot(np.arange(0,len(large)),large)
plt.show()
plt.plot(np.arange(0,len(smaller)),smaller)
plt.show()
large = data[data['Chance of Admit '] >= 0.75]['CGPA']
smaller= data[data['Chance of Admit '] < 0.75]['CGPA']
print(large.describe())
print(smaller.describe())
plt.plot(np.arange(0,len(large)),large)
plt.show()
plt.plot(np.arange(0,len(smaller)),smaller)
plt.show()
data_corr = data.corr()
data_corr
plt.figure(figsize=(10,10))
sns.heatmap(data_corr, annot=True)
plt.show()
train_y = data['Chance of Admit ']
del data['Chance of Admit ']
del data['Research']
train_y
del data['LOR ']
del data['Serial No.']
data
from sklearn.linear_model import LinearRegression
from sklearn import tree
model = LinearRegression()
model.fit(data,train_y)

print(model.score(data,train_y))
new_data = model.predict(data)
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(data,train_y)

print(model.score(data,train_y))
new_data = model.predict(data)
from sklearn.linear_model import Lasso
model = Lasso()
model.fit(data,train_y)

print(model.score(data,train_y))
new_data = model.predict(data)
test = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
test
test_y = test['Chance of Admit ']
del test['Chance of Admit ']
del test['Research']
del test['LOR ']
del test['Serial No.']
model.predict(test)
# Linear Reggression
print(model.score(test,test_y))
model.predict(test)
# Lasso Reggression
print(model.score(test,test_y))