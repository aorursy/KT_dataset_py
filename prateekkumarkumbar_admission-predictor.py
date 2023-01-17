# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.metrics import r2_score
import pickle

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
data.head()
data.describe()
data.drop(['Serial No.'], axis=1)
data.corr()
features = ['GRE Score', 'TOEFL Score', 'University Rating','SOP','LOR ','CGPA', 'Research']
X = data[features].values
y = data['Chance of Admit '].values.reshape(-1,1)
X
y
plt.scatter(data['GRE Score'], data['Chance of Admit '])
plt.scatter(data['TOEFL Score'], data['Chance of Admit '])
plt.scatter(data['CGPA'], data['Chance of Admit '])
plt.scatter(data['University Rating'], data['Chance of Admit '])
plt.scatter(data['LOR '], data['Chance of Admit '])
xx = X.to_numpy()
standardize_x = preprocessing.normalize(X)
standardize_x
X_train, X_test, y_train, y_test = train_test_split(standardize_x,y, test_size=0.3, random_state=2)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
mean_squared_error(y_test,y_pred)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df
D = {'GRE':[301], 'TOEFL':[98], 'University_ranking':[3], 'SOP':[4], 'LOR':[4], 'CGPA':[8.48], 'Research':[1]}
my_score = pd.DataFrame(D)
my_score = preprocessing.normalize(my_score)

D
my_score
model.predict(my_score)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X,y, test_size=0.1, random_state=2)
model1 = LinearRegression()
model1.fit(X_train1, y_train1)
y_pred1 = model1.predict(X_test1)
mean_squared_error(y_test1,y_pred1)
model1.score(X_test1, y_test1)
r2_score(y_test1,y_pred1)
df1 = pd.DataFrame({'Actual': y_test1.flatten(), 'Predicted': y_pred1.flatten()})
df1
e = {'GRE':[300], 'TOEFL':[100], 'University_ranking':[5], 'SOP':[5], 'LOR':[5], 'CGPA':[10.0], 'Research':[1]}
my_score = pd.DataFrame(e)
my_score
res = model1.predict(my_score)
res
filename = 'UAP.pkl'
pickle.dump(model1, open(filename, 'wb'))
