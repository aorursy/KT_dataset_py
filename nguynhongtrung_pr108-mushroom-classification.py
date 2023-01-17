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
# 80% train, and 20% test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
dataset['class'].unique()
classifier1 = KNeighborsClassifier(n_neighbors=2)
classifier1.fit(x_train, y_train) 
y_pred = classifier1.predict(x_test)
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
y_pred = classifier2.predict(x_test)
#Making the confusion matrix 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(cm, annot=True, linewidth=5, cbar=None)
plt.title('SVM with linear kernel Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('predicted label')
print('accuracy of train dataset is',classifier2.score(x_train, y_train))
print('accuracy of test dataset is',classifier2.score(x_test, y_test))


print('accuracy of train dataset is',classifier3.score(x_train, y_train))
print('accuracy of test dataset is',classifier3.score(x_test, y_test))
classifier4 = DecisionTreeClassifier(criterion='gini', max_depth=7, random_state=1)
classifier4.fit(x_train,y_train)
y_pred = classifier4.predict(x_test)
#Making the confusion matrix 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(cm, annot=True, linewidth=5, cbar=None)
plt.title('Decision tree ')
plt.ylabel('True label')
plt.xlabel('predicted label')
print('accuracy of train dataset is',classifier4.score(x_train, y_train))
print('accuracy of test dataset is',classifier4.score(x_test, y_test))
classifier5 = RandomForestClassifier(criterion='gini',
                                n_estimators=1000, 
                                random_state=1)
classifier5.fit(x_train,y_train)
#Predicting the Test set results 
y_pred = classifier5.predict(x_test)
#Making the confusion matrix 
#from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(cm, annot=True, linewidth=5, cbar=None)
plt.title('Random forest confusion matrix')
plt.ylabel('True label')
plt.xlabel('predicted label')
print('accuracy of train set ',classifier5.score(x_train, y_train))
print('accuracy of test set ',classifier5.score(x_test, y_test))
# Standardizing the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1.0, random_state=1)
lr.fit(x_train_std, y_train)

y_pred = lr.predict(x_test_std)


#Compute performance
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
mcm = multilabel_confusion_matrix(y_test, y_pred)
tn = mcm[:, 0, 0]
tp = mcm[:, 1, 1]
fn = mcm[:, 1, 0]
fp = mcm[:, 0, 1]
print('Confusion matrix')
print(confusion_matrix(y_test, y_pred))
print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, linewidth=5, cbar=None)
plt.title('Logistic regression')
plt.ylabel('True label')
plt.xlabel('predicted label')