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



import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn import metrics

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
sample = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
train.shape
train.sample(10)
train.dtypes
train['target'].values
train["ID_code"] = train.index.values
train.head()
train.isnull().any().any()
train.std(axis = 0, skipna = True)
train[train.columns[2:]].std().plot('hist');

plt.title('Distribution of stds of all columns in our dataset');
train[train.columns[2:]].mean().plot('hist');

plt.title('Distribution of means of all columns');
train.corr()
sns.countplot(x='target', data=train)
sns.distplot(train['var_0']);
sns.distplot(train['var_1']);
x = train.drop(columns=['ID_code','target'])

y = train['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
y_train.value_counts()
y_test.value_counts()
classifier = LogisticRegression(solver = 'lbfgs', random_state = 0)
classifier.fit(x_train, y_train)
predicted_y_log = classifier.predict(x_test)
predicted_y_log
for x in range(len(predicted_y_log)):

    if(predicted_y_log[x]==1):

        print(x, end="\t")
print('Accuracy: {:.2f}'.format(classifier.score(x_test, y_test)))
confusion_matrix = confusion_matrix(y_test, predicted_y_log)

print(confusion_matrix)
print(classification_report(y_test, predicted_y_log))
svclassifier = SVC(kernel='linear')

svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("The accuracy of the model: ",metrics.accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
gnb = GaussianNB()
gnb.fit(x_train,y_train)
gnb.score(x_train,y_train)
y_pred_gnb = gnb.predict(x_test)
print(confusion_matrix(y_test, y_pred_gnb))

print(classification_report(y_test, y_pred_gnb))
clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image  

import pydotplus



dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True,feature_names = feature_cols,class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('diabetes.png')

Image(graph.create_png())
print("Classification report:",metrics.classification_report(y_test, y_pred))
from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True, feature_names = feature_cols,class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('diabetes.png')

Image(graph.create_png())
model = RandomForestClassifier(n_estimators=100, 

                               bootstrap = True,

                               max_features = 'sqrt')
model.fit(x_train, y_train)
predictions = model.predict(x_test)
errors = abs(predictions - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
y_pred = model.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)