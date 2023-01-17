# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

sns.set(style="white", color_codes=True)

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/HR_comma_sep.csv')

data.head()
corr = data.corr()

corr = (corr)

corr



sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of Correlation Matrix')



plt.show()
#splitting the dataset manually

train = data[:7499]

test = data[7499:]



#separating the sales and salary independently

sales_dummies_train = pd.get_dummies(train["sales"])

sales_dummies_test = pd.get_dummies(test["sales"])



train.drop(["sales"], axis = 1, inplace=True)

test.drop(["sales"], axis = 1, inplace=True)



train = train.join(sales_dummies_train)

test = test.join(sales_dummies_test)



train.head()

test.head()



salary_dummies_train = pd.get_dummies(train["salary"])

salary_dummies_test = pd.get_dummies(test["salary"])



train = train.join(salary_dummies_train)

test = test.join(salary_dummies_test)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



sns.countplot(x="salary", data=train, ax=axis1, order=['low', 'medium', 'high'])



salary_avg = train[["salary", "left"]].groupby(['salary'], as_index=False).mean()

sns.barplot(x='salary', y='left', data=salary_avg, ax=axis2, order=['low', 'medium', 'high'])

train.drop(["salary"], axis = 1, inplace=True)

test.drop(["salary"], axis = 1, inplace=True)
satisfaction_level=data['satisfaction_level']

last_evaluation=data['last_evaluation']

number_project=data['number_project']

average_montly_hours=data['average_montly_hours']

time_spend_company=data['time_spend_company']

Work_accident=data['Work_accident']



sns.pairplot(data, hue="left", vars=['satisfaction_level', 'last_evaluation', 'average_montly_hours','number_project'])

plt.show()
sns.pairplot(data, hue="salary", vars=['satisfaction_level','time_spend_company', 'last_evaluation', 'average_montly_hours','number_project'])

plt.show()


data_train = train.drop("left", axis=1)

label_train = train["left"]

data_test = test.drop("left", axis=1).copy()

label_test = test["left"]



data_train.info()

data_test.info()
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10,7,10))

mlp.fit(data_train, label_train)

predictions = mlp.predict(data_test)

score = accuracy_score(label_test, predictions)

print('Accuracy score of MLP CLASSIFIER:', score)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(data_train, label_train)

predict = gnb.predict(data_test)

algo0 = accuracy_score(label_test, predict)

print('Accuracy Score of Naive Bayes:', algo0)
from sklearn.linear_model import LogisticRegression

logis = LogisticRegression()

logis.fit(data_train, label_train)

predict = logis.predict(data_test)

algo1 = accuracy_score(label_test, predict)

print('Accuracy Score of Logistic Regression:', algo1)
from sklearn.svm import SVC

svm = SVC()

svm.fit(data_train, label_train)

predict = svm.predict(data_test)

algo2 = accuracy_score(label_test, predict)

print('Accuracy Score of SVC:', algo2)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(data_train, label_train)

predict = knn.predict(data_test)

algo3 = accuracy_score(label_test, predict)

print('Accuracy Score of KNN:', algo3)
from sklearn import tree

from sklearn.tree import export_graphviz

dt = tree.DecisionTreeClassifier(max_depth=3)

dt.fit(data_train, label_train)

predict = dt.predict(data_test)

algo4 = accuracy_score(label_test, predict)

print('Accuracy Score of Decision Tree:', algo4)



from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10)

rfc.fit(data_train, label_train)

prediction = rfc.predict(data_test)

algo5 = accuracy_score(label_test, prediction)

print('Accuracy Score of Random Forest Classifier:', algo5)
print("Since RFC yields highest accuracy. Let's display its classification report")



print(classification_report(label_test, prediction))
models = pd.DataFrame({

        'Model'          : ['MLP Neural Network', "Naive Bayes", 'Logistic Regression', 'SVM', 'kNN', 'Decision Tree', 'Random Forest'],

        'Accuracy_score' : [score, algo0, algo1, algo2, algo3, algo4, algo5]

    })

models.sort_values(by='Accuracy_score', ascending=False)
import itertools

from itertools import product

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

	plt.imshow(cm, interpolation='nearest', cmap=cmap)

	plt.title(title)

	plt.colorbar()

	tick_marks = np.arange(len(classes))

	plt.xticks(tick_marks, classes, rotation=45)

	plt.yticks(tick_marks, classes)

	print('Confusion Matrix')

	print(cm)

	thresh = cm.max()/2

	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        	plt.text(j, i, cm[i, j],

                	horizontalalignment="center",

                	color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()

	plt.ylabel('True label')

	plt.xlabel('Predicted Label')



cnf_matrix = confusion_matrix(label_test, prediction)

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['YES','NO'], title='Confusion matrix, Yes = (Class is 1) & No = (Class is 0)')

plt.show()
prediction_model = rfc



train_imp = train.drop("left", axis=1)



importances = rfc.feature_importances_

std = np.std([tree.feature_importances_ for tree in rfc.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(train_imp.shape[1]):

    print("%d. %s (%f)" % (f + 1, train_imp.columns[indices[f]], importances[indices[f]]))





# Plot the feature importances of the forest

plt.figure(figsize=(10, 5))

plt.title("Feature importances")

plt.bar(range(train_imp.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(train_imp.shape[1]), train_imp.columns[indices], rotation='vertical')

plt.xlim([-1, train_imp.shape[1]])

plt.show()
