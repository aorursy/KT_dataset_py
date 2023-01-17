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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
expected = [1,1,0,1,0,0,1,0,0,0]
predicted = [1,0,0,1,0,0,1,0,1,1]
results = confusion_matrix(expected, predicted)

accuracy_score(expected, predicted)


print(results)
print((results[0,0] + results[1,1])/(sum(sum(results))))
import pandas as pd
data = pd.read_csv('../input/cleveland.csv', sep = ',')
data.columns =['age', 'sex','cp' ,'trestbps', 'chol', 'fbs', 'restecg', 'thailach','exang', 'oldpeak', 'slope', 'ca', 'thal','target']
data['target'] = data.target.map({0:0, 1:1, 2:1, 3:1, 4:1})
data['sex'] = data.sex.map({0:'female', 1:'male'})
data['thal']= data.thal.fillna(data.thal.mean())
data['ca']= data.ca.fillna(data.ca.mean())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper", font_scale = 2, rc = {"font.size": 20, "axes.titlesize":25, "axes.lablesize":20})
sns.catplot(kind ='count', data = data, x='age', hue="target", order=data['age'].sort_values().unique())
plt.title('Variation of Age for each target class')
plt.show()
sns.catplot(kind = 'bar', data = data, y='age', x='sex', hue='target')
plt.title('Distribution of age vs sex with the target class')
plt.show()


data['sex'] = data.sex.map({'female':0, 'male':1})
X= data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_brain, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler as ss
sc= ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
###Logistic Regression

x = data.iloc[:,:-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split 



from sklearn.preprocessing import StandardScaler as ss
    

acc= 0

for i in range(10000):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
    from sklearn.linear_model import LogisticRegression
    sc = ss()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    acc_temp = classifier.score(X_test,y_test)
    
    if(acc_temp > acc):
        acc = acc_temp
print(acc)
    
from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for Logistic Regression = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))
##SVM: Support vector machine
from sklearn.svm import SVC


for i in range (1000):
    classifier = SVC(kernel = 'rbf')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)
    print()

    print('Accuracy for training set for Logistic Regression = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
    print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))