import numpy as np
from numpy import array
from numpy import argmax
import pandas as pd
from sklearn.preprocessing import scale, MinMaxScaler
from keras.utils import to_categorical

import seaborn as sb
import matplotlib.pyplot as plt
import pandas_profiling
from PIL import Image

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam, SGD

from sklearn.metrics import accuracy_score, log_loss, average_precision_score, f1_score, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
data_frame_train = pd.read_csv('../input/titanic/train.csv')
data_frame_test = pd.read_csv('../input/titanic/test.csv')
data_frame_train.head()
print('Total row col of train data : ',data_frame_train.shape[0],' ',data_frame_train.shape[1])
print('Total row col of test data  : ',data_frame_test.shape[0],' ',data_frame_test.shape[1])
data_frame_train.isnull().sum()
data_frame_test.isnull().sum()
data_frame_train.info()
data_frame_train.describe()
data_frame_train['Age'] = data_frame_train['Age'].fillna(data_frame_train['Age'].median())
data_frame_test['Age'] = data_frame_test['Age'].fillna(data_frame_test['Age'].median())
data_frame_test['Fare'] = data_frame_test['Fare'].fillna(0)
data_frame_train.Embarked.value_counts()
data_frame_train['Embarked'] = data_frame_train['Embarked'].fillna('S')
data_frame_test['Embarked'] = data_frame_test['Embarked'].fillna('S')
data_frame_train['Sex'] = pd.factorize(data_frame_train.Sex)[0]
data_frame_test['Sex'] = pd.factorize(data_frame_test.Sex)[0]

data_frame_train['Embarked'] = pd.factorize(data_frame_train.Embarked)[0]
data_frame_test['Embarked'] = pd.factorize(data_frame_test.Embarked)[0]
print('After converting categorical Sex data to numerical      :',data_frame_train.Sex.unique())
print('After converting categorical Embarked data to numerical :',data_frame_train.Embarked.unique())
data_frame_train.head()
data_frame_train.isnull().sum()
data_frame_train['Cabin'].describe()
data_frame_train['Ticket'].describe()
print('Total unique feature in Cabin column :',data_frame_train['Cabin'].nunique())
print('Total unique feature in Ticket column :',data_frame_train['Ticket'].nunique())
data_frame_train = data_frame_train.drop(['Ticket'], axis=1)
data_frame_train = data_frame_train.drop(['Cabin'], axis=1)
data_frame_train = data_frame_train.drop(['Name'], axis=1)
data_frame_test = data_frame_test.drop(['Ticket'], axis=1)
data_frame_test = data_frame_test.drop(['Cabin'], axis=1)
data_frame_test = data_frame_test.drop(['Name'], axis=1)
data_frame_train.info()
data_frame_test['Age'] = data_frame_test['Age'].astype(int)
data_frame_test['Fare'] = data_frame_test['Fare'].astype(int)

data_frame_train['Age'] = data_frame_train['Age'].astype(int)
data_frame_train['Fare'] = data_frame_train['Fare'].astype(int)
data_frame_train.info()
sb.set(rc={'figure.figsize':(10,10)})
sb.heatmap(data_frame_train.corr(), annot = True)
data_frame_test.head()
data_frame_train.head()
sb.set_style("whitegrid");
sb.pairplot(data_frame_train, hue="Survived")

image = Image.open('../input/images/Coorelation_1.png')
image
plt.figure(figsize=(10,8), dpi= 80)
sb.kdeplot(data_frame_train.loc[data_frame_train['Sex'] == 0, "Survived"], shade=True, color="g", label="Male=0")
sb.kdeplot(data_frame_train.loc[data_frame_train['Sex'] == 1, "Survived"], shade=True, color="deeppink", label="Female=1")
# Decoration
plt.title('Survived', fontsize=22)
plt.legend()
plt.show()
image = Image.open('../input/images/kde.png')
image
plt.figure(figsize=(15,15), dpi= 80)
sb.factorplot('Survived', 'Age', data=data_frame_train, hue='Sex')
plt.show()
plt.figure(figsize=(10,8), dpi= 80)
sb.barplot('Survived','Age',data=data_frame_train,hue='Sex')
plt.show()
sb.factorplot('Survived',data=data_frame_train,kind='count',hue='Sex')
plt.show()
sb.factorplot("Pclass", "Survived", hue = "Sex", data = data_frame_train)
plt.show()
pd.crosstab([data_frame_train["Sex"], data_frame_train["Survived"]], data_frame_train["Pclass"], 
            margins = True).style.background_gradient(cmap = "summer_r")
sb.barplot(x = "Sex", y = "Survived", hue = "Pclass", data = data_frame_train)
plt.show()
sb.barplot(x = "Embarked", y = "Survived", hue = "Pclass", data = data_frame_train)
plt.show()
grid = sb.FacetGrid(data_frame_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
#plt.xkcd()
data_frame_train.head()
data_frame_test.head()
drop_feature = ['PassengerId','Survived']
x_train = data_frame_train.drop(drop_feature, axis=1)
y_train = data_frame_train['Survived']
x_test = data_frame_test.drop('PassengerId', axis=1)
x_train.head()
x_test.head()
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
x_train_scaled[0] # Age 80 SibSp 8 Fare 511.999
clf = LogisticRegression()

clf.fit(x_train_scaled, y_train)
y_pred_log_reg = clf.predict(x_test_scaled)
acc_log_reg = round( clf.score(x_train_scaled, y_train) * 100, 2)
print (str(acc_log_reg) + ' %')
clf = SVC()

clf.fit(x_train_scaled, y_train)
y_pred_svc = clf.predict(x_test_scaled)
acc_svc = round(clf.score(x_train_scaled, y_train) * 100, 2)
print (str(acc_svc) + '%')
clf = LinearSVC()

clf.fit(x_train_scaled, y_train)
y_pred_linear_svc = clf.predict(x_test_scaled)
acc_linear_svc = round(clf.score(x_train_scaled, y_train) * 100, 2)
print (str(acc_linear_svc) + '%')
from sklearn.linear_model import SGDClassifier
# sgd = linear_model.SGDClassifier()
sgd = SGDClassifier()

sgd.fit(x_train_scaled, y_train)
Y_pred = sgd.predict(x_test_scaled)
sgd.score(x_train_scaled, y_train)
acc_sgd = round(sgd.score(x_train_scaled, y_train) * 100, 2)
print(str(acc_sgd)+'%')
clf = RandomForestClassifier()

clf.fit(x_train_scaled, y_train)
Y_prediction_randomforest = clf.predict(x_test_scaled)
clf.score(x_train_scaled, y_train)
acc_random_forest = round(clf.score(x_train_scaled, y_train) * 100, 2)
print(str(acc_random_forest) + '%')
clf = KNeighborsClassifier()

clf.fit(x_train_scaled, y_train)
y_pred_knn = clf.predict(x_test_scaled)
acc_knn = round(clf.score(x_train_scaled, y_train) * 100, 2)
print (str(acc_knn)+'%')
clf = DecisionTreeClassifier()

clf.fit(x_train_scaled, y_train)
y_pred_decision_tree = clf.predict(x_test_scaled)
acc_decision_tree = round(clf.score(x_train_scaled, y_train) * 100, 2)
print (str(acc_decision_tree) + '%')
clf = GaussianNB()

clf.fit(x_train_scaled, y_train)
y_pred_gnb = clf.predict(x_test_scaled)
acc_gnb = round(clf.score(x_train_scaled, y_train) * 100, 2)
print (str(acc_gnb) + '%')
clf = DecisionTreeClassifier()

classifier = clf.fit(x_train_scaled, y_train)
Y_prediction_randomforest = clf.predict(x_train_scaled)
acc_random_forest = round(clf.score(x_train_scaled, y_train) * 100, 2)
print(str(acc_random_forest) + '%')

class_names = ['Survived', 'Not Survived']
title = 'Confusion Matrix'
np.set_printoptions(precision=3)

disp = plot_confusion_matrix(classifier, x_train_scaled, y_train, display_labels=class_names, cmap=plt.cm.Blues)
plt.grid(False)
disp.ax_.set_title(title)
print(title)
print(disp.confusion_matrix)
plt.show()
model = Sequential()
model.add(Dense(4, input_shape=(7,), activation='relu'))
model.add(Dense(1,  activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=.001), metrics=['accuracy'])
model.summary()
history = model.fit(x_train_scaled, y_train, epochs=1000) # y_test_bin
#history = model.fit(x_train, y_train_bin, epochs=1000, validation_data=[x_test, y_test_bin])
#result = model.evaluate(x_test, y_test_bin)
model = Sequential()
model.add(Dense(1, input_shape=(7,), activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=.001), metrics=['accuracy'])
model.summary()
history = model.fit(x_train_scaled, y_train, epochs=1000) # y_test_bin
#result = model.evaluate(x_test, y_test_bin)
classifiers_set_1 = [
    LinearSVC(),
    SGDClassifier(),
    ]

for clf in classifiers_set_1:

  name = clf.__class__.__name__
  clf.fit(x_train_scaled, y_train)

  y_pred_decision_tree = clf.predict(x_train_scaled)
  acc = accuracy_score(y_train, y_pred_decision_tree)
  print('{:<25}'.format(name),": ", " Accuracy: {:.2%}".format(acc))
classifiers_set_2 = [
    LogisticRegression(),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    ]

acc_loss = pd.DataFrame(columns=["Classifier", "Accuracy", "Log Loss"])

print('Accuracy and Loss for Train data in different Classifier : \n')

for clf in classifiers_set_2:
  name = clf.__class__.__name__
  clf.fit(x_train_scaled, y_train)

  y_pred_decision_tree = clf.predict(x_train_scaled)
  acc = accuracy_score(y_train, y_pred_decision_tree)

  y_pred_decision_tree = clf.predict_proba(x_train_scaled)
  loss = log_loss(y_train, y_pred_decision_tree)

  print('{:<25}'.format(name),": ", " Accuracy: {:.2%}".format(acc)," Loss: {:.1}".format(loss))
  
  temp = pd.DataFrame([[name, acc*100, loss]], columns=["Classifier", "Accuracy", "Log Loss"])
  acc_loss = acc_loss.append(temp)

# For ANN
ann_res = model.evaluate(x_train_scaled, y_train,steps=None)
print('{:<25}'.format('ANN'),": ", " Accuracy: {:.2%}".format(ann_res[1])," Loss: {:.1}".format(ann_res[0]))
temp = pd.DataFrame([['ANN', ann_res[1]*100, ann_res[0]]], columns=["Classifier", "Accuracy", "Log Loss"])
acc_loss = acc_loss.append(temp)
acc_loss
plt.figure(figsize=(8,5), dpi= 80)
sb.barplot(x='Accuracy', y='Classifier', data=acc_loss, color="g")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy of Train')
plt.show()
plt.figure(figsize=(8,5), dpi= 80)
sb.barplot(x='Log Loss', y='Classifier', data=acc_loss, color="r")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss of Train')
plt.show()
pred_res = pd.read_csv('../input/titanic/gender_submission.csv')
y_test = pred_res['Survived']
all_id = pred_res['PassengerId']
classifiers_set_3 = [
    LogisticRegression(),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    ]

acc_loss_2 = pd.DataFrame(columns=["Classifier", "Accuracy", "Log Loss"])
print('Accuracy and Loss for Test data in different Classifier : \n')
for clf in classifiers_set_3:
  name = clf.__class__.__name__
  clf.fit(x_train_scaled, y_train)

  y_pred_decision_tree = clf.predict(x_test_scaled)
  acc = accuracy_score(y_test, y_pred_decision_tree)

  y_pred_decision_tree = clf.predict_proba(x_test_scaled)
  loss = log_loss(y_test, y_pred_decision_tree)

  print('{:<25}'.format(name),": ", " Accuracy: {:.2%}".format(acc)," Loss: {:.1}".format(loss))
  
  temp = pd.DataFrame([[name, acc*100, loss]], columns=["Classifier", "Accuracy", "Log Loss"])
  acc_loss_2 = acc_loss_2.append(temp)

# For ANN
#ann_res = model.evaluate(x_train_scaled, y_train,steps=None)
#temp = pd.DataFrame([['ANN', ann_res[1]*100, ann_res[0]]], columns=["Classifier", "Accuracy", "Log Loss"])
#acc_loss = acc_loss.append(temp)
plt.figure(figsize=(8,5), dpi= 80)
sb.barplot(x='Accuracy', y='Classifier', data=acc_loss_2, color="g")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy of Test')
plt.show()
plt.figure(figsize=(8,5), dpi= 80)
sb.barplot(x='Log Loss', y='Classifier', data=acc_loss_2, color="r")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss of Test')
plt.show()
clf = SVC(kernel="rbf", C=0.025, probability=True)

classifier = clf.fit(x_train_scaled, y_train)
y_pred_decision_tree = clf.predict(x_test_scaled)
acc_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
print ("Accuracy : {:.3%}".format(acc_decision_tree))

class_names = ['Survived', 'Not Survived']
title = 'Confusion Matrix'
#np.set_printoptions(precision=2)

disp = plot_confusion_matrix(classifier, x_test_scaled, y_test, display_labels=class_names, cmap=plt.cm.Blues)
plt.grid(False)
disp.ax_.set_title(title)
print(title)
print(disp.confusion_matrix)
clf = SVC(kernel="rbf", C=0.025, probability=True)

classifier = clf.fit(x_train_scaled, y_train)
y_pred_decision_tree = clf.predict(x_test_scaled)
acc_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
print ("Accuracy : {:.3%}".format(acc_decision_tree))
#cd /content/drive/My Drive/Google Colab/Data_Science/Titanic
import pickle
# now you can save it to a file
with open('../input/titanic-99/Titanic_99.pkl', 'wb') as f:
    pickle.dump(clf, f)
# and later you can load it
with open('../input/titanic-99/Titanic_99.pkl', 'rb') as f:
    save_model = pickle.load(f)
save_model.predict(x_test_scaled)
print('F1 macro    : ',f1_score(y_test, y_pred_decision_tree, average='macro'))
print('F1 micro    : ',f1_score(y_test, y_pred_decision_tree, average='micro'))
print('F1 weighted : ',f1_score(y_test, y_pred_decision_tree, average='weighted'))
print('F1 None     : ',f1_score(y_test, y_pred_decision_tree, average=None))
average_precision = average_precision_score(y_test, y_pred_decision_tree)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))
disp = plot_precision_recall_curve(classifier, x_test_scaled, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: ' 'AP={0:0.2f}'.format(average_precision))
Final_Result = pd.DataFrame(list(zip(all_id, y_pred_decision_tree)),columns =['PassengerId', 'Survived']) 
Final_Result.head()
Final_Result.to_csv('../input/output/Final_Result.csv',index=False)
