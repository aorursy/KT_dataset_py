# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/bird.csv")
df.head(3)
df.skew()
#Columnwise null value count

df.isnull().sum()
#fill up nan value with mean of the column

df=df.fillna(df.mean())
#check nan value

df.isnull().sum()
plt.hist(df['huml'])
sns.pairplot(df)
corr=df.corr()

corr = (corr)

plt.figure(figsize=(14,14))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws=



{'size': 15},

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of Correlation Matrix')
df.head(3)
sns.countplot(df['type'])
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

df['type']=encoder.fit_transform(df['type'])
df.head(3)
level=df['type']

df=df.drop('type',axis=1)
from sklearn.model_selection import train_test_split

data_train, data_test, label_train, label_test = train_test_split(df, level, 

test_size = 0.2, random_state = 42)
data_test.shape,data_train.shape
#Logistic Regression

from sklearn.linear_model import LogisticRegression

logis = LogisticRegression()

logis.fit(data_train, label_train)

logis_score_train = logis.score(data_train, label_train)

print("Training score: ",logis_score_train)

logis_score_test = logis.score(data_test, label_test)

print("Testing score: ",logis_score_test)
coeff_df = pd.DataFrame(df.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Correlation"] = pd.Series(logis.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
#decision tree

from sklearn import tree

dt = tree.DecisionTreeClassifier()

dt.fit(data_train, label_train)

dt_score_train = dt.score(data_train, label_train)

print("Training score: ",dt_score_train)

dt_score_test = dt.score(data_test, label_test)

print("Testing score: ",dt_score_test)
#decision tree

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(data_train, label_train)

rf_score_train = rf.score(data_train, label_train)

print("Training score: ",rf_score_train)

rf_score_test = dt.score(data_test, label_test)

print("Testing score: ",rf_score_test)
#sns.barplot(x='index',y='Importance',data=importance)
#kNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(data_train, label_train)

knn_score_train = knn.score(data_train, label_train)

print("Training score: ",knn_score_train)

knn_score_test = knn.score(data_test, label_test)

print("Testing score: ",knn_score_test)
#SVM

from sklearn.svm import SVC

svm = SVC()

svm.fit(data_train, label_train)

svm_score_train = svm.score(data_train, label_train)

print("Training score: ",svm_score_train)

svm_score_test = svm.score(data_test, label_test)

print("Testing score: ",svm_score_test)
#Model comparison

models = pd.DataFrame({

        'Model'          : ['Logistic Regression', 'SVM', 'kNN', 'Decision Tree', 'Random Forest'],

        'Training_Score' : [logis_score_train, svm_score_train, knn_score_train, dt_score_train, rf_score_train],

        'Testing_Score'  : [logis_score_test, svm_score_test, knn_score_test, dt_score_test, rf_score_test]

    })

models.sort_values(by='Testing_Score', ascending=False)
sns.barplot(x='Model',y='Testing_Score',data=models)
sns.pointplot(x='Model',y='Testing_Score',data=models)
sns.pointplot(x='Model',y='Training_Score',data=models)