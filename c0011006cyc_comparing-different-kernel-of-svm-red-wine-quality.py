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



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

df.head(5)
#checking NaN value

df.info()
df['quality'].value_counts()
#Change the type of target to object type

bins = (2,5.5,8)

group_names = ['bad','good']

categories = pd.cut(df['quality'], bins, labels = group_names)

df['quality'] = categories
df['quality'].value_counts()
X = df.drop('quality', axis=1)

y=df['quality']
#encode y(target)

from sklearn.preprocessing import LabelEncoder

le_y = LabelEncoder()

y = le_y.fit_transform(y)

y
#train/test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#construct a dict to store accuracies of different each model

svm_acc = {}
from sklearn.svm import SVC

svm_rbf = SVC(kernel='rbf', random_state=0)

svm_rbf.fit(X_train, y_train)

acc_svm_rbf = svm_rbf.score(X_test, y_test)*100

svm_acc['svm with rbf kernel'] = acc_svm_rbf

print("Test Accuracy of rbf kernel SVM:{:.2f}%".format(acc_svm_rbf))
from sklearn.metrics import confusion_matrix

y_hat_rbf = svm_rbf.predict(X_test)



cm_rbf=confusion_matrix(y_test, y_hat_rbf)

sns.heatmap(cm_rbf, annot=True, fmt='d', cmap="YlGnBu")
print("false predicting by rbf kernel:", cm_rbf[1,0]+cm_rbf[0,1])
#Linear kernel svm

svm_li = SVC(kernel='linear', random_state=0)

svm_li.fit(X_train, y_train)

acc_svm_li = svm_li.score(X_test, y_test)*100

svm_acc['svm with linear kernel'] = acc_svm_li

print("Test Accuracy of linear kernel SVM:{:.2f}%".format(acc_svm_li))
y_hat_li = svm_li.predict(X_test)



cm_li=confusion_matrix(y_test, y_hat_li)

sns.heatmap(cm_li, annot=True, fmt='d', cmap="YlGnBu")
print("false predicting by linear kernel:", cm_li[1,0]+cm_li[0,1])
#Polynomial kernel SVM

svm_poly = SVC(kernel='poly', random_state=0)

svm_poly.fit(X_train, y_train)

acc_svm_poly = svm_poly.score(X_test, y_test)*100

svm_acc['svm with ploy kernel'] = acc_svm_poly

print("Test Accuracy of polynomial kernel SVM:{:.2f}%".format(acc_svm_poly))
y_hat_poly = svm_poly.predict(X_test)



cm_poly=confusion_matrix(y_test, y_hat_poly)

sns.heatmap(cm_poly, annot=True, fmt='d', cmap="YlGnBu")
print("false predicting by polynomial kernel:", cm_poly[1,0]+cm_poly[0,1])
#sigmoid kernel svm

svm_sig = SVC(kernel='sigmoid', random_state=0)

svm_sig.fit(X_train, y_train)

acc_svm_sig = svm_sig.score(X_test, y_test)*100

svm_acc['svm with sigmoid kernel'] = acc_svm_sig

print("Test Accuracy of sigmoid kernel SVM:{:.2f}%".format(acc_svm_sig))
y_hat_sig = svm_sig.predict(X_test)



cm_sig=confusion_matrix(y_test, y_hat_sig)

sns.heatmap(cm_sig, annot=True, fmt='d', cmap="YlGnBu")
print("false predicting by sigmoid kernel:", cm_sig[1,0]+cm_sig[0,1])
#transform dict svm_acc to a dataframe

acc_svm_df = pd.DataFrame(svm_acc.items(), columns=['kernel', 'acc_score'])

acc_svm_df.head()
#Visulize accuracy score for each kerel

plt.figure(figsize=(18,8))

ax = sns.barplot(x='kernel', y='acc_score', data = acc_svm_df)