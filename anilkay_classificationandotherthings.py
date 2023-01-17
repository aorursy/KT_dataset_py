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
data=pd.read_csv("/kaggle/input/malware-analysis-datasets-pe-section-headers/pe_section_headers.csv")

data.tail()
howmanyna=data.isna().sum().sum()

print(howmanyna)
x=data.iloc[:,1:5]

y=data.iloc[:,5:]
%matplotlib inline

import seaborn as sns

sns.countplot(data=data,x="malware")
sns.relplot(data=data,x="size_of_data",y="virtual_size",hue="malware")
sizedifference=data["virtual_size"]-data["size_of_data"]

sizedifference
print(sizedifference.max())

print(sizedifference.min())
newx=pd.DataFrame({"sizedif":sizedifference,"entr":data["entropy"]})

newx.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

rfc=RandomForestClassifier()

#scores = cross_val_score(clf, iris.data, iris.target, cv=5)

cross_val_score(rfc,newx,y,cv=10)
rfc=RandomForestClassifier()

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(rfc,newx, y, cv=10)

conf_mat = confusion_matrix(y, y_pred)
conf_mat
import warnings  

warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier

rfc=DecisionTreeClassifier()

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(rfc,newx, y, cv=10)

conf_mat = confusion_matrix(y, y_pred)

print(conf_mat)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

scale_newx=sc.fit_transform(newx)
from sklearn.svm import SVC

svm=SVC(kernel="rbf")

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(svm,scale_newx, y, cv=10)

conf_mat = confusion_matrix(y, y_pred)

print(conf_mat)
from sklearn.svm import SVC

svm=SVC(kernel="linear")

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(svm,scale_newx, y, cv=10)

conf_mat = confusion_matrix(y, y_pred)

print(conf_mat)
rfc=RandomForestClassifier()

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(rfc,x, y, cv=10)

conf_mat = confusion_matrix(y, y_pred)

print(conf_mat)
from sklearn.tree import DecisionTreeClassifier

rfc=DecisionTreeClassifier()

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(rfc,x, y, cv=10)

conf_mat = confusion_matrix(y, y_pred)

print(conf_mat)
newplotx=pd.DataFrame({"sizedif":sizedifference,"entr":data["entropy"],"mal":y["malware"]})

newplotx.head()
sns.relplot(data=newplotx,x="sizedif",y="entr",hue="mal")
from sklearn.naive_bayes import GaussianNB

gbc=GaussianNB()

y_pred = cross_val_predict(gbc,newx, y, cv=10)

conf_mat = confusion_matrix(y, y_pred)

print(conf_mat)
from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=0)

X_resampled, y_resampled = cc.fit_resample(x, y)
len(X_resamled)
from sklearn.svm import SVC

svm=SVC(kernel="rbf")

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(svm,X_resampled, y_resampled, cv=10)

conf_mat = confusion_matrix(y_resampled, y_pred)

print(conf_mat)
rfc=RandomForestClassifier()

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(rfc,X_resampled, y_resampled, cv=10)

conf_mat = confusion_matrix(y_resampled, y_pred)

print(conf_mat)
rfc=DecisionTreeClassifier()

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(rfc,X_resampled, y_resampled, cv=10)

conf_mat = confusion_matrix(y_resampled, y_pred)

print(conf_mat)
from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=0)

X2_resampled, y2_resampled = cc.fit_resample(newx, y)
rfc=RandomForestClassifier()

y_pred = cross_val_predict(rfc,X2_resampled, y2_resampled, cv=10)

conf_mat = confusion_matrix(y2_resampled, y_pred)

print(conf_mat)
rfc=DecisionTreeClassifier()

y_pred = cross_val_predict(rfc,X2_resampled, y2_resampled, cv=10)

conf_mat = confusion_matrix(y2_resampled, y_pred)

print(conf_mat)