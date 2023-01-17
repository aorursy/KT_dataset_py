# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import seaborn as sns

from matplotlib import pyplot as plt



import warnings as wrn

wrn.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing Data Using Pandas

data = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data.head()
data.tail()
data.info()
data["class"] = [1 if each == "Normal" else 0 for each in data["class"]] 
data.head()
data.tail()
data = (data-np.min(data)) /(np.max(data)-np.min(data))
data.head()

from sklearn.model_selection import train_test_split

x = data.drop("class",axis=1)

y = data["class"]



x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,test_size=0.3)
from sklearn.metrics import confusion_matrix

def plot_confusionMatrix(y_true,y_pred):

    cn = confusion_matrix(y_true=y_true,y_pred=y_pred)

    

    fig,ax = plt.subplots(figsize=(5,5))

    sns.heatmap(cn,annot=True,linewidths=1.5)

    plt.show()

    return cn
score_list = {} # I've created this dict for saving score variables into it 
from sklearn.neighbors import KNeighborsClassifier 

KNN = KNeighborsClassifier(n_neighbors=22) #I've tried more than 50 values. 22 is the best value



KNN.fit(x_train,y_train)

knn_score = KNN.score(x_test,y_test)

score_list["KNN Classifier"] = knn_score

print(f"Score is {knn_score}")

y_true = y_test

y_pred = KNN.predict(x_test)

plot_confusionMatrix(y_true,y_pred)
from sklearn.linear_model import LogisticRegression



LR = LogisticRegression()

LR.fit(x_train,y_train)



lr_score = LR.score(x_test,y_test)

score_list["Logistic Regression"] = lr_score



print(f"Score is {lr_score}")
y_pred = LR.predict(x_test)

plot_confusionMatrix(y_true,y_pred)
from sklearn.svm import SVC 



svc = SVC()

svc.fit(x_train,y_train)

svc_score = svc.score(x_test,y_test)

score_list["SVC"] = svc_score

print(f"Score is {svc_score}")
y_true = y_test

y_pred = svc.predict(x_test)

plot_confusionMatrix(y_true,y_pred)
from sklearn.naive_bayes import GaussianNB



nbc = GaussianNB()

nbc.fit(x_train,y_train)

nbc_score = nbc.score(x_test,y_test)

score_list["GaussianNBC"] = nbc_score



print(f"Score is {nbc_score}")
y_true = y_test

y_pred = nbc.predict(x_test)

plot_confusionMatrix(y_true,y_pred)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=1)

dtc.fit(x_train,y_train)



dtc_score = dtc.score(x_test,y_test)

score_list["DTC"] = dtc_score

print(f"Score is {dtc_score}")
y_true = y_test

y_pred = dtc.predict(x_test)

plot_confusionMatrix(y_true,y_pred)
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=50,random_state=1)

rfc.fit(x_train,y_train)

rfc_score = rfc.score(x_test,y_test)

score_list["RFC"]=rfc_score



print(f"Score is {rfc_score}")
y_true = y_test

y_pred = rfc.predict(x_test)

plot_confusionMatrix(y_true,y_pred)
score_list = list(score_list.items())
for alg,score in score_list:

    print(f"{alg} Score is {str(score)[:4]} ")
