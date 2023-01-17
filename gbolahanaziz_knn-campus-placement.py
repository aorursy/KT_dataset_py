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
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
plt.style.use('ggplot')
sns.set_style('darkgrid')
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head(5)
data.describe()
data.info()
sns.pairplot(data,hue='specialisation')
sns.countplot(data['status'],hue=data['specialisation'])
sns.boxplot('status','degree_p',data=data,hue='specialisation')
sns.countplot(data['specialisation'])
sns.scatterplot('hsc_p','degree_p',hue='status',data=data)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
Gender= pd.get_dummies(data['gender'],drop_first=True)
Sec =  pd.get_dummies(data['ssc_b'],drop_first=True)
Hsc_b =  pd.get_dummies(data['hsc_b'],drop_first=True)
Hsc_s =  pd.get_dummies(data['hsc_s'],drop_first=True)
Degree =  pd.get_dummies(data['degree_t'],drop_first=True)
Work =  pd.get_dummies(data['workex'],drop_first=True)
Spec =  pd.get_dummies(data['specialisation'],drop_first=True)
new_data = pd.concat([data,Spec,Work,Degree,Hsc_s,Sec,Gender,Hsc_b],axis=1)
new_data = new_data.drop(['sl_no','gender','ssc_b','hsc_b','hsc_s','workex','degree_t','specialisation'],axis=1)
new_data.head(4)
X=new_data.drop(['status','salary'],axis=1)
y=new_data['status']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test,pred))
print('\n')
print(accuracy_score(y_test,pred))
error_rate = []


for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K_Value')

