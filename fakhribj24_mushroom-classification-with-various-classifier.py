import pandas as pd 

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.svm import SVC 

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier 
dataset = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

dataset.head()
dataset.columns
dataset.info()
encoding = LabelEncoder()

for i in dataset.columns:

    dataset[i] = encoding.fit_transform(dataset[i])
dataset.head()
dataset["stalk-root"].value_counts()
dataset.corr()
dataset = dataset.drop('veil-type', axis=1)
x = dataset.drop(['class'], axis=1)  #delete target column from train dataset

y = dataset['class'] # test dataset  
# divide dataset into 65% train, and other 35% test.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=0)
dataset['class'].unique()
classifier1 = KNeighborsClassifier(n_neighbors=2)

classifier1.fit(x_train, y_train)

#Predicting the Test set results 

y_pred = classifier1.predict(x_test)

#Making the confusion matrix 

cm = confusion_matrix(y_test,y_pred)

print(cm)
sns.heatmap(cm, annot=True, linewidth=5, cbar=None)

plt.title('KNN Classifier confusion matrix')

plt.ylabel('True label')

plt.xlabel('predicted label')
print('accuracy of train dataset is',classifier1.score(x_train, y_train))

print('accuracy of test dataset is',classifier1.score(x_test, y_test))
classifier2 = SVC(kernel = 'linear', random_state = 0)

classifier2.fit(x_train, y_train)

#Predicting the Test set results 

y_pred = classifier2.predict(x_test)

#Making the confusion matrix 

#from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test,y_pred)

print(cm)
sns.heatmap(cm, annot=True, linewidth=5, cbar=None)

plt.title('SVM with linear kernel Classifier confusion matrix')

plt.ylabel('True label')

plt.xlabel('predicted label')
print('accuracy of train dataset is',classifier2.score(x_train, y_train))

print('accuracy of test dataset is',classifier2.score(x_test, y_test))
classifier3 = SVC(kernel = 'rbf', random_state = 0)

classifier3.fit(x_train, y_train)

#Predicting the Test set results 

y_pred = classifier3.predict(x_test)

#Making the confusion matrix 

#from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test,y_pred)

print(cm)
sns.heatmap(cm, annot=True, linewidth=5, cbar=None)

plt.title('SVM with rbf kernel Classifier confusion matrix')

plt.ylabel('True label')

plt.xlabel('predicted label')
print('accuracy of train dataset is',classifier3.score(x_train, y_train))

print('accuracy of test dataset is',classifier3.score(x_test, y_test))
classifier4 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier4.fit(x_train, y_train)

#Predicting the Test set results 

y_pred = classifier4.predict(x_test)

#Making the confusion matrix 

#from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test,y_pred)

print(cm)
sns.heatmap(cm, annot=True, linewidth=5, cbar=None)

plt.title('Decision tree with entropy impurity confusion matrix')

plt.ylabel('True label')

plt.xlabel('predicted label')
print('accuracy of train dataset is',classifier4.score(x_train, y_train))

print('accuracy of test dataset is',classifier4.score(x_test, y_test))
classifier5 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier5.fit(x_train,y_train)

#Predicting the Test set results 

y_pred = classifier5.predict(x_test)

#Making the confusion matrix 

#from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test,y_pred)

print(cm)
sns.heatmap(cm, annot=True, linewidth=5, cbar=None)

plt.title('RF with with entropy impurity Classifier confusion matrix')

plt.ylabel('True label')

plt.xlabel('predicted label')
print('accuracy of train dataset is',classifier5.score(x_train, y_train))

print('accuracy of test dataset is',classifier5.score(x_test, y_test))