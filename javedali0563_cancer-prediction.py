from sklearn import datasets

from sklearn import svm

import sklearn

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from IPython.display import display,HTML



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

cancer = datasets.load_breast_cancer()
cancer.feature_names
cancer.target_names
x = cancer.data

y = cancer.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y , test_size = 0.1)
classes = ['malignant', 'benign']
clf = svm.SVC(kernel = 'linear')

clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
acc = metrics.accuracy_score(y_test,y_predict)

print(acc)
html="<table>"

html+="<tr>"

html+="<td>Actual Value</td><td>Predicted Value</td>"

html+="</tr>"

for i in range(len(y_predict)):

    html+="<tr>"

    html+="<td>%s</td>"%(classes[y_test[i]])

    html+="<td>%s</td>"%(classes[y_predict[i]])

    html+="</tr>"

display(HTML(html))