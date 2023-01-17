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

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current sessio
#Reading the data

data = pd.read_excel('/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')

data.head()
data.shape
data.columns.value_counts()
data.dtypes
data.describe()
data.isnull().sum()
import seaborn as sns

sns.heatmap(data.isnull())
for i in data.columns:

    if type(data[i].iloc[0]) == str:

        factor = pd.factorize(data[i])

        data[i] = factor[0]

        definitions = factor[1]
from sklearn.model_selection import train_test_split

#Independent Vector

X = data[list(data.columns)[:-1]].values

#Dependent Vector

y = data[data.columns[-1]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify=y)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(np.nan_to_num(X_train))

X_test = scaler.transform(np.nan_to_num(X_test))
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_fscore_support
# Create Decision Tree classifer object

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)

#Checing accuracy of model

acc =  metrics.accuracy_score(y_test, y_pred)
from sklearn.metrics import roc_curve, auc

print('accuracy ' +str(acc))

#print('average auc ' +str(roc_auc["average"]))

prfs = precision_recall_fscore_support(y_test, y_pred, labels = [0,1])

fpr, tpr, _ = roc_curve(y_test, y_pred)

roc_auc = auc(fpr, tpr)

print('precision:',prfs[0] )

print('recall', prfs[1])

print('fscore', prfs[2])
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_fscore_support

model = RandomForestClassifier(n_jobs=64,n_estimators=200,criterion='entropy',oob_score=True)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc =  metrics.accuracy_score(y_test, y_pred)
print('accuracy ' +str(acc))

#print('average auc ' +str(roc_auc["average"]))

prfs = precision_recall_fscore_support(y_test, y_pred, labels = [0,1])

fpr, tpr, _ = roc_curve(y_test, y_pred)

roc_auc = auc(fpr, tpr)

print('precision:',prfs[0] )

print('recall', prfs[1])

print('fscore', prfs[2])
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression(max_iter=1000)

logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

acc =  metrics.accuracy_score(y_test, y_pred)
print('accuracy ' +str(acc))

#print('average auc ' +str(roc_auc["average"]))

prfs = precision_recall_fscore_support(y_test, y_pred, labels = [0,1])

fpr, tpr, _ = roc_curve(y_test, y_pred)

roc_auc = auc(fpr, tpr)

print('precision:',prfs[0] )

print('recall', prfs[1])

print('fscore', prfs[2])
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix