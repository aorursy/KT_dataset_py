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
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data_y = pd.read_csv('../input/data0611-1/data_csv_0608_y_notime_c5.csv')
data_x_raw = pd.read_csv('../input//data0611/data_csv_0608_x_notime_cp_density.csv')
print(data_y.shape)
print(list(data_y.columns))
print(list(data_x_raw.columns))
#data_x=data_x_raw.drop(columns=['cp_bem_015_d','cp_bem_011_d','cp_bem_013_d'])
data_x=data_x_raw.drop(columns=['cp_bem_015_d','cp_bem_013_d']) #cluster2
#data_x=data_x_raw
#data_x=data_x_raw.drop(columns=['cp_bem_018_d','cp_bem_010_d','cp_bem_017_d','cp_bem_019_d','cp_bem_008_d']) #cluster2

scaler = StandardScaler()
data_x_scaled=scaler.fit_transform(data_x)
data_x=pd.DataFrame(data_x_scaled,columns=list(data_x.columns))

y_field="cluster_2"

#data_x_sel=data_x.loc[(data_y[y_field] == 2) | (data_y[y_field] == 3)| (data_y[y_field] == 1)| (data_y[y_field] == 5)]
#data_y_sel=data_y.loc[(data_y[y_field] == 2) | (data_y[y_field] == 3)| (data_y[y_field] == 1)| (data_y[y_field] == 5)]

data_x_sel=data_x.loc[(data_y[y_field] == 2) | (data_y[y_field] == 3)]
data_y_sel=data_y.loc[(data_y[y_field] == 2) | (data_y[y_field] == 3)]

#cluster_2 =2
#cluster_1 =1

train_x, test_x, train_y, test_y  = train_test_split(data_x_sel,data_y_sel[y_field],random_state=2) 
C = 10
kernel = 'rbf'
gamma  = 0.01

svm = SVC(C=C, kernel=kernel, gamma=gamma,probability=True).fit(train_x, train_y)
pred_y_svm = svm.predict(test_x)

print(y_field+'_SVM: {:.5f}'.format(accuracy_score(test_y, pred_y_svm)))
print(y_field+'_SVM: {:.5f}'.format(svm.score(train_x, train_y)))
print(y_field+'_SVM: {:.5f}'.format(svm.score(test_x, test_y)))
import shap
shap.initjs()
explainer = shap.KernelExplainer(svm.predict_proba, train_x)
shap_values = explainer.shap_values(test_x)
shap.summary_plot(shap_values, train_x)
