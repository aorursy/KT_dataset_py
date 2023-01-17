#define of the libraries we need
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#data preprocessing
data = pd.read_csv("../input/column_2C_weka.csv")
data['class'] = [1 if item=='Abnormal' else 0 for item in data['class']]
y = data['class'].values
xdata = data.drop(['class'],axis=True)
#normlization
x = (xdata - np.min(xdata)) / (np.max(xdata) - np.min(xdata))
#create train and test data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 1)
#accuracy list for evalute model each other
classifier = ['knn','svm','lr','nb','dt','rf']
accuracy_list = []
# create knn model
scorelist = {}
for item in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=item)
    knn.fit(x_train, y_train)
    scorelist[item] = knn.score(x_test,y_test)
acc = max(zip(scorelist.values(),scorelist.keys()))
accuracy_list.append(acc[0])
#create support vector machines model
svm = SVC(random_state=1)
svm.fit(x_train, y_train)
svm.score(x_test, y_test)
accuracy_list.append(svm.score(x_test, y_test))
#create Logistic Regression model
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr.score(x_test, y_test)
accuracy_list.append(lr.score(x_test, y_test))
#create naive bayes model
nb = GaussianNB()
nb.fit(x_train,y_train)
nb.score(x_test, y_test)
accuracy_list.append(nb.score(x_test, y_test))
#create decision tree model
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)
dtree.score(x_test, y_test)
accuracy_list.append(dtree.score(x_test, y_test))
#create random forest model
scorelist = {}
for item in range(100,1000,10):
    rf = RandomForestClassifier(n_estimators=101, random_state=1)
    rf.fit(x_train, y_train)
    scorelist[item] = rf.score(x_test,y_test)
acc = max(zip(scorelist.values(),scorelist.keys()))
accuracy_list.append(acc[0])
#accuracy visualization
bar = plt.bar(classifier,accuracy_list)
plt.yticks(np.arange(0, max([0, 0.5, 1]), 0.05))
plt.xlabel('Classifier')
plt.ylabel('Value')
plt.show()
