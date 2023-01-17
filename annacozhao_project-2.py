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
train_data = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")
train_data = train_data.iloc[:,1:]
train_data.shape
train_data.dtypes
train_data.skew()
train_data['Cover_Type'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt

data = train_data.iloc[:,:10] 
cols = data.columns 
data_corr = data.corr()
corr_list = []

for i in range(0,10): 
    for j in range(i+1,10): 
        if (data_corr.iloc[i,j] >= 0.5 and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -0.5):
            corr_list.append([data_corr.iloc[i,j],i,j]) 
        
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

for v,i,j in s_corr_list:
    sns.pairplot(train_data, hue = "Cover_Type", size = 6, x_vars = cols[i],y_vars = cols[j] )
    plt.show()
cols = dataset.columns

size = len(cols)-1

x = cols[size]

y = cols[0:size]

for i in range(0,size):
    sns.violinplot(data=dataset,x=x,y=y[i])  
    plt.show()

train_data = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/forest-cover-type-prediction/test.csv")
test_data.head()
x = train_data.drop(['Id','Cover_Type'],axis=1)
y = train_data['Cover_Type']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=40)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_jobs = -1, n_neighbors = 1)
knn.fit(x_train, y_train)
acc_knn_result = knn.score(x_test, y_test)
print(acc_knn_result)
from sklearn.svm import SVC

for i in [10]:
    #Set the base model
    svcm = SVC(random_state=0,C=i)

svcm.fit(x_train, y_train)
acc_svm_result = svcm.score(x_test, y_test)
print(acc_svm_result)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=70)
rfc.fit(x_train,y_train)
acc_rfc_result = rfc.score(x_test, y_test)
print(acc_rfc_result)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
acc_gnb_result = gnb.score(x_test, y_test)
print(acc_gnb_result)
from sklearn.tree import DecisionTreeClassifier
for max_depth in [13]:
    dtc = DecisionTreeClassifier(random_state = 0,max_depth = max_depth)
dtc.fit(x_train,y_train)
acc_dtc_result = dtc.score(x_test, y_test)
print(acc_dtc_result)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

basement = DecisionTreeClassifier(random_state = 0,max_depth = 13)
n_list = [100]

for i in n_list:
    bdt = BaggingClassifier(n_jobs = -1,base_estimator = basement, n_estimators = i, random_state = 0)
bdt.fit(x_train,y_train)
acc_bdt_result = bdt.score(x_test, y_test)
print(acc_bdt_result)
from sklearn.ensemble import RandomForestClassifier

n_list = [100]

for i in n_list:
    rfc = RandomForestClassifier(n_jobs = -1, n_estimators = i, random_state = 0)

rfc.fit(x_train,y_train)
acc_rfc_result = rfc.score(x_test, y_test)
print(acc_rfc_result)
from sklearn.ensemble import ExtraTreesClassifier

for i in [100]:
    etc = ExtraTreesClassifier(n_jobs=-1,n_estimators=i, random_state=0)
etc.fit(x_train,y_train)
acc_etc_result = etc.score(x_test, y_test)
print(acc_etc_result)
dataset_test = test_data.drop("Id", axis = 1)
predict = rfc.predict(dataset_test)
submission = pd.DataFrame(data = predict,columns = ['Cover_Type'])
submission["Id"] = test_data["Id"]
submission.set_index("Id",inplace = True)
submission.head()
submission.to_csv("Submission.csv")