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



from collections import Counter

from sklearn import svm

from sklearn import metrics

from sklearn.model_selection import train_test_split



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder() #for x encoding

le = LabelEncoder() #for y encoding according to manual



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('/kaggle/input/jrc-warnings-agricultural-vegetation-anomalies/warnings-ts-zip/warnings_ts.csv',

                   encoding="windows-1251")

data.head()
data.describe()
df = data.loc[:, ["asap0_id","asap0_name","w_crop","w_crop_na","w_crop_de","w_crop_gr"]]

df.head()
sns.violinplot(x=df["w_crop"])

plt.show()

#As I see, there are values from 0-20 and from 80 to 99.
#Let's look at all values we have

rate = Counter(df["w_crop"])

rate
warning_des = Counter(df["w_crop_de"])

warning_des
warning_na = Counter(df["w_crop_na"])

warning_na
warning_gr = Counter(df["w_crop_gr"])

warning_gr
df = df.drop(df[df.w_crop == 99].index)

df.describe()
X_enc = enc.fit_transform(df.loc[:,["w_crop_gr","w_crop_na"]])

enc.categories_
X_enc
plt.rcParams["figure.figsize"] = (12,7)

sns.countplot(df.iloc[:,2])

plt.show()
#Preparing data

dataX = X_enc

dataY = df.loc[:,["w_crop"]]
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size = 0.3, random_state = 42)

X_train_ = np.c_[np.ones(len(X_train)), X_train]

X_test_ = np.c_[np.ones(len(X_test)), X_test]



y_train = y_train.to_numpy()

y_train = y_train.ravel() #flattern

y_test = y_test.to_numpy()

y_test = y_test.ravel()
model_svm = svm.SVC(C=1,kernel="rbf",gamma="scale")

model_svm.fit(X_train, y_train)

pred_svm_tr = model_svm.predict(X_train)

pred_svm_te = model_svm.predict(X_test)

acc_svm_tr = metrics.accuracy_score(pred_svm_tr, y_train)

acc_svm_te = metrics.accuracy_score(pred_svm_te, y_test)

print(acc_svm_tr, acc_svm_te)
model_knn = KNeighborsClassifier(n_neighbors=1)

model_knn.fit(X_train, y_train)

knn_pred_tr = model_knn.predict(X_train)

knn_pred_te = model_knn.predict(X_test)

knn_acc_tr = metrics.accuracy_score(knn_pred_tr, y_train)

knn_acc_te = metrics.accuracy_score(knn_pred_te, y_test)

print(knn_acc_tr, knn_acc_te)
rfc = RandomForestClassifier(n_estimators=10, max_features=None) #default params (100 - n_estimators)

rfc.fit(X_train, y_train)

rfc_pred_tr = rfc.predict(X_train)

rfc_pred_te = rfc.predict(X_test)

rfc_acc_tr = metrics.accuracy_score(rfc_pred_tr, y_train)

rfc_acc_te = metrics.accuracy_score(rfc_pred_te, y_test)

print(knn_acc_tr, knn_acc_te)