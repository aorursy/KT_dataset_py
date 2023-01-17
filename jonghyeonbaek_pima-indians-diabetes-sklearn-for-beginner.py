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
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
data
data.info()
data.isnull().sum()
Death_Preg = data[data['Outcome'] == 1]['Pregnancies']
Alive_Preg = data[data['Outcome'] == 0]['Pregnancies']
print(Death_Preg.describe())
print(Alive_Preg.describe())
plt.plot(np.arange(0,len(Death_Preg)),Death_Preg)
plt.show()
plt.plot(np.arange(0,len(Alive_Preg)),Alive_Preg)
plt.show()
Death_BloodPressure= data[data['Outcome'] == 1]['BloodPressure']
Alive_BloodPressure = data[data['Outcome'] == 0]['BloodPressure']
print(Death_BloodPressure.describe())
print(Alive_BloodPressure.describe())
plt.plot(np.arange(0,len(Death_BloodPressure)),Death_BloodPressure)
plt.show()
plt.plot(np.arange(0,len(Alive_BloodPressure)),Alive_BloodPressure)
plt.show()
Death_Glucose = data[data['Outcome'] == 1]['Glucose']
Alive_Glucose = data[data['Outcome'] == 0]['Glucose']
print(Death_Glucose.describe())
print(Alive_Glucose.describe())
plt.plot(np.arange(0,len(Death_Glucose)),Death_Glucose)
plt.show()
plt.plot(np.arange(0,len(Alive_Glucose)),Alive_Glucose)
plt.show()
Death_SkinThickness = data[data['Outcome'] == 1]['SkinThickness']
Alive_SkinThickness = data[data['Outcome'] == 0]['SkinThickness']
print(Death_SkinThickness.describe())
print(Alive_SkinThickness.describe())
plt.plot(np.arange(0,len(Death_SkinThickness)),Death_SkinThickness)
plt.show()
plt.plot(np.arange(0,len(Alive_SkinThickness)),Alive_SkinThickness)
plt.show()
Death_Insulin = data[data['Outcome'] == 1]['Insulin']
Alive_Insulin = data[data['Outcome'] == 0]['Insulin']
print(Death_Insulin.describe())
print(Alive_Insulin.describe())
plt.plot(np.arange(0,len(Death_Insulin)),Death_Insulin)
plt.show()
plt.plot(np.arange(0,len(Alive_Insulin)),Alive_Insulin)
plt.show()
Death_BMI = data[data['Outcome'] == 1]['BMI']
Alive_BMI = data[data['Outcome'] == 0]['BMI']
print(Death_BMI.describe())
print(Alive_BMI.describe())
plt.plot(np.arange(0,len(Death_BMI)),Death_BMI)
plt.show()
plt.plot(np.arange(0,len(Alive_BMI)),Alive_BMI)
plt.show()
Death_DiabetesPedigreeFunction = data[data['Outcome'] == 1]['DiabetesPedigreeFunction']
Alive_DiabetesPedigreeFunction = data[data['Outcome'] == 0]['DiabetesPedigreeFunction']
print(Death_DiabetesPedigreeFunction.describe())
print(Alive_DiabetesPedigreeFunction.describe())
plt.plot(np.arange(0,len(Death_DiabetesPedigreeFunction)),Death_DiabetesPedigreeFunction)
plt.show()
plt.plot(np.arange(0,len(Alive_DiabetesPedigreeFunction)),Alive_DiabetesPedigreeFunction)
plt.show()
Death_Age = data[data['Outcome'] == 1]['Age']
Alive_Age = data[data['Outcome'] == 0]['Age']
print(Death_Age.describe())
print(Alive_Age.describe())
plt.plot(np.arange(0,len(Death_Age)),Death_Age)
plt.show()
plt.plot(np.arange(0,len(Alive_Age)),Alive_Age)
plt.show()
data_corr = data.corr()
data_corr
plt.figure(figsize=(10,10))
sns.heatmap(data_corr,annot=True)
plt.show()
data_y = data['Outcome']
del data['Outcome']
del data['SkinThickness']
del data['Glucose']
# 제가 모르고 2번을 눌러서 ;;; 한번 누르면 잘 됩니다.
from sklearn.linear_model import LogisticRegression
from sklearn import tree
model = LogisticRegression()
model.fit(data,data_y)

new_data = model.predict(data)
print(model.score(data,data_y))
print(new_data)
print(data_y)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data,data_y)

print(clf.score(data,data_y))
new_data_2 = clf.predict(data)
print(new_data_2)

count = 0
for i in range(len(new_data)):
  if new_data_2[i] != data_y[i]:
      count = count + 1
print(count)   