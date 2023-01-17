# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
data = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
#As Always the first few rows
data.head()
data.describe()
data.isnull().sum()
#There seems no null values, which make EDA much simpler
#Lets see the corelation between the vaues, but the serial no column has no signifiance, 
#so we will drop Serial no
data.drop(['Serial No.'],inplace=True,axis=1)
data.head()
data.corr()
corr = data.corr()
sns.heatmap(corr,cmap="Blues",annot=True)
#Chance of Admit is the Value to be predicted
#Hatmap shows that GRE Score,TOEFL Score,CGPA has high impact on Chance of Admit
df = data.drop(['Chance of Admit '],axis=1)
y = data['Chance of Admit ']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
from sklearn.linear_model import LinearRegression
log_reg=LinearRegression()
log_reg.fit(X_train,y_train)
pred = log_reg.predict(X_test)
from sklearn.metrics import r2_score
print (r2_score(y_test, pred))
print (pred)
