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
data = pd.read_csv("/kaggle/input/hr-analytics-case-study/general_data.csv")

data.head(5)
data.shape
data.columns
from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()

data['BusinessTravel'] = labelEncoder_X.fit_transform(data['BusinessTravel'])

data['Department'] = labelEncoder_X.fit_transform(data['Department'])

data['EducationField'] = labelEncoder_X.fit_transform(data['EducationField'])

data['Gender'] = labelEncoder_X.fit_transform(data['Gender'])

data['JobRole'] = labelEncoder_X.fit_transform(data['JobRole'])

data['MaritalStatus'] = labelEncoder_X.fit_transform(data['MaritalStatus'])

data['Over18'] = labelEncoder_X.fit_transform(data['Over18'])
from sklearn.preprocessing import LabelEncoder

label_encoder_y = LabelEncoder()

data['Attrition'] = label_encoder_y.fit_transform(data['Attrition'])
data.head()
data.isnull().any()
import math

mean_companies_worked = math.floor(data["NumCompaniesWorked"].mean())

data["NumCompaniesWorked"].fillna(mean_companies_worked, inplace = True)

mean_working_years = math.floor(data["TotalWorkingYears"].mean())

data["TotalWorkingYears"].fillna(mean_working_years, inplace = True)
data.isnull().any()
corr = data.corr()

print(corr)

import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize = (18, 9))

sns.heatmap(corr, annot = True, linewidth = 0.05, cmap = 'BuPu')

plt.show()
X = data[['Age', 'EducationField', 'Gender', 'JobLevel',

          'JobRole', 'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',

       'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',

       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',

       'YearsWithCurrManager']]

y = data["Attrition"]
X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from sklearn.preprocessing import StandardScaler

Scaler_X = StandardScaler()

X_train = Scaler_X.fit_transform(X_train)

X_test = Scaler_X.transform(X_test)

len(y_test)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)
model.predict(X_test)
predicted_y = model.predict(X_test)
model.score(X_test, y_test)
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 



results = confusion_matrix(predicted_y, y_test) 

  

print('Confusion Matrix :')

print(results) 

print("Accuracy Score: ", accuracy_score(y_test, predicted_y))

print("Classification Report: \n", classification_report(y_test, predicted_y))
