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
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
data.head()
data.shape
data.isna().sum()
data.dtypes
import missingno as msno 
msno.matrix(data)
data = data.drop(['sl_no'],axis =1)
for i in data.columns:
    print(f'length of unique values in {i}',len(set(data[i])))
    print(f'some of the unique values in {i}',list(set(data[i]))[0:5])
    print('---------------------------------------------------------')
data.loc[data['gender'] =='M', 'gender'] = 1
data.loc[data['gender'] =='F', 'gender'] = 0

for i in ['hsc_b','ssc_b']:
    data.loc[data[i] =='Central', i] = 1
    data.loc[data[i] =='Others', i] = 0

data.loc[data['hsc_s'] =='Arts', 'hsc_s'] = 0
data.loc[data['hsc_s'] =='Science', 'hsc_s'] = 1
data.loc[data['hsc_s'] =='Commerce', 'hsc_s'] = 2

data.loc[data['degree_t'] =='Comm&Mgmt', 'degree_t'] = 0
data.loc[data['degree_t'] =='Sci&Tech', 'degree_t'] = 1
data.loc[data['degree_t'] =='Others', 'degree_t'] = 2

data.loc[data['workex'] =='Yes', 'workex'] = 1
data.loc[data['workex'] =='No', 'workex'] = 0


data.loc[data['status'] =='Placed', 'status'] = 1
data.loc[data['status'] =='Not Placed', 'status'] = 0

data.loc[data['specialisation'] =='Mkt&Fin', 'specialisation'] = 1
data.loc[data['specialisation'] =='Mkt&HR', 'specialisation'] = 0

 

for i in data.columns:
    data[i] = pd.to_numeric(data[i],errors='coerce')

data.dtypes
data.isna().sum()
data1 = data.dropna(axis=0,subset =['salary','status'])

missing_data = data1.loc[:,['salary','status']]
missing_salary = data['salary'].isnull()
status_for_missing_values = pd.DataFrame(data['status'][missing_salary])
set(list(status_for_missing_values.status))
    
data['salary'].fillna(value=0, inplace=True)
data.isna().sum()
f, ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.drop(['status'],axis =1).corr(), annot=True ,linewidth=0.5, fmt='.1f',ax=ax);
plt.figure(figsize=(20,20))
columns = ['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation']
j =0
for i in columns:
    j +=1
    plt.subplot(2,7,j)
    sns.barplot(x= i , y=data['status'], data = data)
    plt.title(f"status vs {i}")
plt.show()
train_data = data[:175]
test_data = data[175:]
train_data.head()

y = train_data["status"]
y_test = test_data["status"]

features = ["workex","ssc_p","degree_p","hsc_p","specialisation",'etest_p','mba_p']

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

model = LogisticRegression(penalty= 'none' ,random_state=42 ,max_iter=150).fit(X, y)
y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score
plot_confusion_matrix(model, X_test, y_test,labels=[0,1],normalize= 'true')
print(classification_report(y_test, y_pred, labels=[0,1]))
data.status.value_counts()
