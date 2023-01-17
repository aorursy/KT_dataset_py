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
df_train = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv',header=None)

df_test = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv',header=None)

df_labels = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv',header=None)

print(df_train.shape,df_test.shape,df_labels.shape)

print('--------')

#print(df_labels)
df_train.head(5)
df_labels.head(5)
print(df_train.describe())#描述数据

print(df_train.dtypes) #查看数据类型
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df_train,df_labels,random_state=1)

print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train,y_train)

lr.score(X_test,y_test)
from sklearn.tree import DecisionTreeClassifier



tr = DecisionTreeClassifier().fit(X_train,y_train)

tr.score(X_test,y_test)
from sklearn.svm import SVC

svm = SVC().fit(X_train,y_train)

svm.score(X_test,y_test)
from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB().fit(X_train,y_train)

gnb.score(X_test,y_test)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 



lda = LinearDiscriminantAnalysis().fit(X_train,y_train)

lda.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier().fit(X_train,y_train)

knn.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier().fit(X_train,y_train)

rf.score(X_test,y_test)
from xgboost import XGBClassifier



xgb = XGBClassifier().fit(X_train,y_train)

xgb.score(X_test,y_test)
import matplotlib.pyplot as plt

%matplotlib inline



def score(model):

    return model.score(X_test,y_test)

    

x = ['lr','svm','gnb','knn','lda','tree','rf','xgb']

y = [score(lr),score(svm),score(gnb),score(knn),score(lda),score(tr),score(rf),score(xgb)]



plt.barh(x,y)

plt.xlim(0.6,1.0)

plt.show()
model = SVC().fit(df_train,df_labels)

pred = model.predict(df_test)
print(pred.shape)

pred[0:10]
submit = pd.DataFrame({

    'Id':range(1,9001),

    'Solution':pred

}

)

submit.head(5)
submit.to_csv('submit.csv',index=False)