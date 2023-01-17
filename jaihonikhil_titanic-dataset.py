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
import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
# import pandas as pd
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
data = pd.read_csv("../input/titanic/train_and_test2.csv")


#Visualização inicial do dataset para primeira análise
data.shape
# data=data.drop(columns=['Passengerid', 'zero', 'zero.1',
#        'zero.2', 'zero.3', 'zero.4', 'zero.5', 'zero.6', 'zero.7',
#        'zero.8', 'zero.9', 'zero.10', 'zero.11', 'zero.12', 'zero.13',
#        'zero.14', 'zero.15', 'zero.16', 'zero.17',
#        'zero.18'])
data.describe()
data.shape
to_drop=['Passengerid', 'zero', 'zero.1',
       'zero.2', 'zero.3', 'zero.4', 'zero.5', 'zero.6', 'zero.7',
       'zero.8', 'zero.9', 'zero.10', 'zero.11', 'zero.12', 'zero.13',
       'zero.14', 'zero.15', 'zero.16', 'zero.17',
       'zero.18']
data1=data.drop(columns=to_drop)
# data.set_index(columns=to_drop, inplace=True) would have changed the data in the place. Made the drop changes in the original i


total = data1.isnull().sum().sort_values(ascending=False)
percent_1 = data1.isnull().sum()/data1.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head
# we see embarked has 2 missing values which should be filled with most frequent values
data1.info()
data1['Embarked'].describe()
common_value = 2
# data = [train_df, test_df]

# for dataset in data:
data1['Embarked'] = data1['Embarked'].fillna(common_value)
data1.loc[2]
# data2=data1.set_index('Age') in case the index was not mentioned specifically
data1.get_dtype_counts()
# we can also rename the columns like
name={'2urvived':'Survived'}

data1.rename(columns=name,inplace=True)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

X_train, X_test = train_test_split(data1, test_size=.33)

Y_train=X_train["Survived"]
X_train = X_train.drop('Survived', axis=1)
Y_test=X_test["Survived"]
X_test = X_test.drop('Survived', axis=1)
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_test.shape
Y_pred_1 = logreg.predict(X_test)
acc_log = logreg.score(X_train,Y_train) * 100
acc_log
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_2 = svc.predict(X_test)
acc_linear_svc = svc.score(X_train,Y_train) * 100
# acc_svc
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, Y_train)
Y_pred_3 = knn.predict(X_test)
acc_knn = knn.score(X_train, Y_train) * 100
acc_knn
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_4 = decision_tree.predict(X_test)
acc_decision_tree = decision_tree.score(X_train, Y_train) * 100
acc_decision_tree


random_forest = RandomForestClassifier(n_estimators=50)
random_forest.fit(X_train,Y_train)
Y_pred_5= random_forest.predict(X_test)
acc_random_forest = random_forest.score(X_train,Y_train) * 100
acc_rf
from sklearn.linear_model import Perceptron
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
gaussian = GaussianNB() 
gaussian.fit(X_train, Y_train)  
Y_pred = gaussian.predict(X_test)  
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)


X=data1.values
X = scale(X)
pca = PCA(n_components=4)
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
results = pd.DataFrame({'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression','Random Forest', 'Naive Bayes', 'Perceptron','Stochastic Gradient Decent','Decision Tree'],'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron,acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=5, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head






importances.plot.bar()
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)
from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(Y_train, predictions))
print("Recall:",recall_score(Y_train, predictions))
from sklearn.metrics import f1_score
f1_score(Y_train, predictions)
from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(Y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()

def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1.5, 0, 1.5])

plt.figure(figsize=(14, 7))
plot_precision_vs_recall(precision, recall)
plt.show()

from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()
from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(Y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)
# pca.fit(X)

# #The amount of variance that each PC explains
# var= pca.explained_variance_ratio_

# #Cumulative Variance explains
# var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
# plt.plot(var1)
# pca.fit(X)
# X1=pca.fit_transform(X)
# plt.plot(X1)
