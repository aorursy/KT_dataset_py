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
data = pd.read_csv("../input/employee-dataset/People Charm case.csv")
data
data.shape
data.info()
data.isnull().sum()
sns.heatmap(data.isnull())
data['dept'].unique()
data['dept'].nunique()
data['dept'].value_counts()
data['salary'].unique()
data['salary'].value_counts()
data['satisfactoryLevel'].value_counts()
data['numberOfProjects'].value_counts()
sns.boxplot(data['avgMonthlyHours'])
sns.boxplot(data['satisfactoryLevel'])
sns.boxplot(data['lastEvaluation'])
sns.distplot(data["avgMonthlyHours"])
sns.distplot(data["lastEvaluation"])
numerical_features = ['satisfactoryLevel','lastEvaluation','numberOfProjects','avgMonthlyHours','timeSpent.company']

categorical_features = ['dept','salary','workAccident','promotionInLast5years']
print(data[numerical_features].hist(bins=15, figsize=(15, 6), layout=(2, 4)))
sns.countplot(data['dept'])
sns.countplot(data['salary'])
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4, figsize=(20, 8))
for variable, subplot in zip(categorical_features, ax.flatten()):
    sns.countplot(data[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
import matplotlib.pyplot as plt
plt.figure(figsize = (15,10))
sns.boxplot(x="salary",y="timeSpent.company",data=data)   #boxplot
plt.xticks(rotation=90)
plt.figure(figsize = (15,10))
sns.boxplot(x="salary",y="avgMonthlyHours",data=data)   #boxplot
plt.xticks(rotation=90)
data.head()
from sklearn.preprocessing import LabelEncoder
x1= LabelEncoder()
data['salary'] = x1.fit_transform(data['salary'])
data.head()
data['salary'].nunique()
data.head()
data['dept'] = x1.fit_transform(data['dept'])
data.head(3)
# importing libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
X = data.drop(['left'],axis=1)   # independent variables
X.head()
Y = data["left"]          # dependent variables
Y.head()
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=3)
y_test.shape
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
print("Confusion Matrix: ",confusion_matrix(y_test,y_pred),sep='\n')
print("Accuracy Score: ",accuracy_score(y_test, y_pred)*100)
from sklearn import metrics
probs = rf.predict_proba(x_test)
prob_positive = probs[:,1]
fpr,tpr,threshold = metrics.roc_curve(y_test,prob_positive)
roc_auc = metrics.auc(fpr,tpr)
print('Area under the curve:',roc_auc)
plt.title('Reciever Operating characterstics')
plt.plot(fpr, tpr,'Orange',label='AUC= %0.2f'%roc_auc)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1],'r--')


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()
