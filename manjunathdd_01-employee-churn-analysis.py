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
#Importing the necessary libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
HR = pd.read_csv("../input/hr-analytics/HR_comma_sep.csv")
HR.head()
HR.describe()
HR.info()
HR.dtypes
HR.isnull().sum()
HR.shape
HR.left.value_counts()
sns.countplot(x="left", data=HR, palette="bwr")

plt.show()
Notleft = len(HR[HR.left == 1])

left = len(HR[HR.left == 0])

print("Percentage of Emplyee Haven't Left the Org: {:.2f}%".format((Notleft / (len(HR.left))*100)))

print("Percentage of Emplyee Left the Org: {:.2f}%".format((left / (len(HR.left))*100)))
sns.countplot(x='promotion_last_5years', data=HR, palette="mako_r")

plt.xlabel("promotion_last_5years (0 = Not Promoted, 1= promoted)")

plt.show()
HR.groupby('left').mean()
pd.crosstab(HR['time_spend_company'],HR.left).plot(kind="bar",figsize=(20,6))

plt.title('left the company as time spend by emp')

plt.xlabel('time_spend_company')

plt.ylabel('Frequency')

plt.savefig('leftandtimespent.png')

plt.show()
pd.crosstab(HR.salary,HR.left).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])

plt.title('Left the org Frequency for Salary')

plt.xlabel('Salary (0 = low, 1 = medium, 2 = high)')

plt.xticks(rotation=0)

plt.legend(["Left the org", "Not left the org"])

plt.ylabel('Frequency')

plt.show()
pd.crosstab(HR['number_project'],HR.left).plot(kind="bar",figsize=(20,6))

plt.title('left the company as no of projects completed by emp')

plt.xlabel('number_project')

plt.ylabel('Frequency')

plt.savefig('leftandnoofprojects.png')

plt.show()
pd.crosstab(HR['satisfaction_level'],HR.left).plot(kind="bar",figsize=(24,6),color=['#DAF7A6','#FF5733' ])

plt.title('Employee left Frequency for Satisfaction level')

plt.xlabel('The emp satisfaction level with rate ')

plt.ylabel('Frequency')

plt.savefig('leftAndsatisfactionlevel.png')

plt.show()
sns.pairplot(HR,hue='left')
sns.pointplot(y="satisfaction_level", x="left", data=HR)
a = pd.get_dummies(HR['Department'], prefix = "Department")

b = pd.get_dummies(HR['salary'], prefix = "salary")
frames = [HR, a, b]

HR = pd.concat(frames, axis = 1)

HR.head()
HR = HR.drop(columns = ['Department', 'salary'])

HR.head()
y = HR.left.values

x_data = HR.drop(['left'], axis = 1)
# Normalize

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
#transpose matrices

x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T
accuracies = {}



lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

acc = lr.score(x_test.T,y_test.T)*100



accuracies['Logistic Regression'] = acc

print("Test Accuracy {:.2f}%".format(acc))
# KNN Model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k

knn.fit(x_train.T, y_train.T)

prediction = knn.predict(x_test.T)



print("{} NN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))
from sklearn.svm import SVC
svm = SVC(random_state = 1)

svm.fit(x_train.T, y_train.T)



acc = svm.score(x_test.T,y_test.T)*100

accuracies['SVM'] = acc

print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train.T, y_train.T)



acc = nb.score(x_test.T,y_test.T)*100

accuracies['Naive Bayes'] = acc

print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train.T, y_train.T)



acc = dtc.score(x_test.T, y_test.T)*100

accuracies['Decision Tree'] = acc

print("Decision Tree Test Accuracy {:.2f}%".format(acc))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(x_train.T, y_train.T)



acc = rf.score(x_test.T,y_test.T)*100

accuracies['Random Forest'] = acc

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))
# Comparing Models

colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]



sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)

plt.show()
# Predicted values

y_head_lr = lr.predict(x_test.T)

knn3 = KNeighborsClassifier(n_neighbors = 3)

knn3.fit(x_train.T, y_train.T)

y_head_knn = knn3.predict(x_test.T)

y_head_svm = svm.predict(x_test.T)

y_head_nb = nb.predict(x_test.T)

y_head_dtc = dtc.predict(x_test.T)

y_head_rf = rf.predict(x_test.T)
from sklearn.metrics import confusion_matrix



cm_lr = confusion_matrix(y_test,y_head_lr)

cm_knn = confusion_matrix(y_test,y_head_knn)

cm_svm = confusion_matrix(y_test,y_head_svm)

cm_nb = confusion_matrix(y_test,y_head_nb)

cm_dtc = confusion_matrix(y_test,y_head_dtc)

cm_rf = confusion_matrix(y_test,y_head_rf)
plt.figure(figsize=(24,12))



plt.suptitle("Confusion Matrixes",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



plt.subplot(2,3,1)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,2)

plt.title("K Nearest Neighbors Confusion Matrix")

sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,3)

plt.title("Support Vector Machine Confusion Matrix")

sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,4)

plt.title("Naive Bayes Confusion Matrix")

sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,5)

plt.title("Decision Tree Classifier Confusion Matrix")

sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,6)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.show()