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
data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
data.head()
data.shape
data.isnull().sum()
data.columns
data.plot.scatter('GRE Score','Chance of Admit ')
data.plot.scatter('TOEFL Score','Chance of Admit ')
data.plot.scatter('University Rating','Chance of Admit ')
data.plot.scatter( 'SOP','Chance of Admit ')
data.plot.scatter('LOR ','Chance of Admit ')
data.plot.scatter( 'CGPA','Chance of Admit ')

data.columns
X = data.drop(columns=['Serial No.','Chance of Admit ','SOP'] , axis = 1)
y = data[['Chance of Admit ']]
#Scaling all the Input features


from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

X_scaled= scale.fit_transform(X)
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=1)
import statsmodels.api as sm

X_train= sm.add_constant(X_train)

model = sm.OLS(y_train,X_train).fit()
model.summary()
X_test = sm.add_constant(X_test)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score

mean_squared_error(y_pred,y_test)

r2_score(y_pred,y_test)
