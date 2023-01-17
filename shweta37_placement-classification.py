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
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head()
data['salary'].fillna(0, inplace = True)
data.isna().sum()
data1 = data.copy(deep=True)
data2 = data.copy(deep=True)
category_col = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in category_col:
    data1[col] = label_encoder.fit_transform(data1[col])

data1.head()
data.head()
pd.crosstab(index=data1['gender'],columns='count') # 1=male, 0=female
pd.crosstab(index=data1['ssc_b'],columns='count') # 1=others, 0=central
pd.crosstab(index=data1['hsc_b'],columns='count') # 1=others, 0=central
pd.crosstab(index=data1['hsc_s'],columns='count') # 2=Science, 1=Commerce, 0=Arts
pd.crosstab(index=data1['degree_t'],columns='count') # 2=Sci&Tech, 1=Others, 0=Comm&Mgmt
pd.crosstab(index=data1['specialisation'],columns='count') # 1=Mkt&HR, 0=Mkt&Fin
pd.crosstab(index=data1['status'],columns='count') # 1=Placed, 0=Not placed
plt.figure(figsize=(12,12))
correlation = data1.corr()
sns.heatmap(correlation, annot=True, cmap='Blues')

#del data1['sl_no']
columns_list=list(data1.columns) #storing col names
print(columns_list)
features=list(set(columns_list)-set(['status'])) #separating output val from data
print(features)
X=data1[features]
y=data1['status']
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
from sklearn.model_selection import train_test_split 
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestClassifier 

model1=RandomForestClassifier(n_estimators=100)

model1.fit(train_X,train_y)

prediction=model1.predict(test_X)
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy:",accuracy_score(test_y, prediction))
cf_matrix = confusion_matrix(test_y,prediction)
sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt="d") # 1=Placed, 0=Not placed