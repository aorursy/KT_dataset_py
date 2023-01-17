

from sklearn.utils import shuffle

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

plt.style.use('ggplot')

plt.style.use('seaborn-darkgrid')
heart_data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

heart_data.columns

class_names = ['Sano', 'Enfermo']
heart_data.head()
# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Distribution of age")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.distplot(a=heart_data['age'], kde=False)



# Add label for vertical axis

plt.ylabel("Number of patients")

plt.grid()

#sns.set_style("dark")
plt.figure(figsize=(10,6))

sns.scatterplot(x=heart_data['age'], y=heart_data['thalach'], hue=heart_data['target'])

plt.grid()

#sns.set_style("dark")
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))







sns.swarmplot(x=heart_data['target'],

              y=heart_data['thalach'], ax = axes[0,0])



sns.swarmplot(x=heart_data['target'],

              y=heart_data['trestbps'], ax = axes[0,1])



sns.swarmplot(x=heart_data['target'],

              y=heart_data['chol'],ax = axes[0,2])



sns.swarmplot(x=heart_data['target'],

              y=heart_data['age'], ax = axes[1,0])



sns.swarmplot(x=heart_data['target'],

              y=heart_data['oldpeak'], ax = axes[1,1])



sns.swarmplot(x=heart_data['target'],

              y=heart_data['cp'], ax = axes[1,2])

#sns.set_style("dark")
feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',

       'exang', 'oldpeak', 'slope', 'ca', 'thal']

train_X, val_X, train_y, val_y = train_test_split(heart_data[feature_cols], heart_data['target'],test_size = 0.15, random_state=1)

casos_train = sum(train_y == 1)

casos_val = sum(val_y == 1)

casos_val







labels = ['Sano', 'Emfermo']

train = [len(train_X)-casos_train , casos_train]

valid = [len(val_X)-casos_val , casos_val]



x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10,5))

rects1 = ax.bar(x - width/2, train, width, label='Train')

rects2 = ax.bar(x + width/2, valid, width, label='Validation')

# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Number of examples')

ax.set_title('Number of Examples: Train and Validation dataset')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()

from sklearn.ensemble import RandomForestClassifier

accuracy_list = {}

rf = RandomForestClassifier(n_estimators=2000, random_state=1).fit(train_X, train_y)



y_pred = rf.predict(val_X)

acc = accuracy_score(val_y, y_pred)*100

accuracy_list['Random Forest'] = acc

print("Accuracy of Random Forest: {:.2f}%".format(acc))



from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(train_X, train_y)

acc = nb.score(val_X, val_y)*100

accuracy_list['GaussianNB'] = acc

print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(train_X, train_y)



acc = svm.score(val_X, val_y)*100

accuracy_list['SVM'] = acc



print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))
# KNN Model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 13)  # n_neighbors means k

knn.fit(train_X, train_y)

prediction = knn.score(val_X, val_y)*100

accuracy_list['KNN'] = prediction

print("KNN Algorithm Accuracy Score : {:.2f}%".format(prediction))

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import plot_confusion_matrix





mlp = MLPClassifier(hidden_layer_sizes=(10,20,10),max_iter=1000, random_state=1)

mlp.fit(train_X,train_y)

predictions = mlp.predict(val_X)

prediction = mlp.score(val_X, val_y)*100

accuracy_list['MLP'] = prediction

print("MLP Algorithm Accuracy Score : {:.2f}%".format(prediction))



from numpy import loadtxt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# fit model no training data

XGB = XGBClassifier()

XGB.fit(train_X,train_y)



predictions = XGB.predict(val_X)

prediction = XGB.score(val_X, val_y)*100



accuracy_list['XGB'] = prediction



print("XGBClassifier Algorithm Accuracy Score : {:.2f}%".format(prediction))

from sklearn.tree import DecisionTreeClassifier



dtree = DecisionTreeClassifier()

dtree.fit(train_X, train_y)



predictions = dtree.predict(val_X)

prediction = dtree.score(val_X, val_y)*100



accuracy_list['Decision Tree'] = prediction



print("Decision Tree Algorithm Accuracy Score : {:.2f}%".format(prediction))

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier(loss = 'modified_huber', shuffle = True, random_state=1)

sgd.fit(train_X, train_y)



predictions = sgd.predict(val_X)

prediction = sgd.score(val_X, val_y)*100



accuracy_list['SGD'] = prediction



print("Stochastic Gradient Descent Accuracy Score : {:.2f}%".format(prediction))

from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(max_iter = 3000)

lr.fit(train_X, train_y)



predictions = lr.predict(val_X)

prediction = lr.score(val_X, val_y)*100



accuracy_list['Logistic Regression'] = prediction



print("Logistic Regression Accuracy Score : {:.2f}%".format(prediction))

accuracy_list
import matplotlib.pyplot as plt



D = {u'Label1':26, u'Label2': 17, u'Label3':30}



plt.bar(range(len(accuracy_list)), list(accuracy_list.values()), align='center')

plt.xticks(range(len(accuracy_list)), list(accuracy_list.keys()))

plt.ylabel('Acuuracy %')

plt.xlabel('Algorithm')

plt.xticks(rotation=75)

plt.show()

#plt.style.use('ggplot')
classifiers = [ mlp, svm, rf, nb, knn, XGB, lr, sgd, dtree]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,10))



for cls, ax in zip(classifiers, axes.flatten()):

    plot_confusion_matrix(cls, 

                          val_X, 

                          val_y, 

                          ax=ax, 

                          cmap='Blues',

                         display_labels=class_names)

    ax.title.set_text(type(cls).__name__)

plt.tight_layout()  

plt.show()

#plt.style.use('ggplot')