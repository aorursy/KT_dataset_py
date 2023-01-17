# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

import warnings

warnings.filterwarnings('always') 

filepath = "../input/phishing-website-detector/phishing.csv"

data = pd.read_csv(filepath)

data.head()
data.info()
data.isnull().sum()
X = data.drop(columns="class")

X
Y=data["class"]

Y=pd.DataFrame(Y)

Y.head()
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=0.3,random_state=2)
print(train_X.shape)

print(train_Y.shape)

print(test_X.shape)

print(test_Y.shape)
corr = data.corr()

fig,ax= plt.subplots(figsize=(20,20))

sns.heatmap(corr,annot=True,linewidth=2.5,ax=ax)


lg=LogisticRegression()

model1=lg.fit(train_X,train_Y)

lg_predict = lg.predict(test_X)

acc_lg=accuracy_score(lg_predict,test_Y)

print(acc_lg)

print(classification_report(lg_predict,test_Y))

con  = confusion_matrix(lg_predict,test_Y)

sns.heatmap(con,annot=True,fmt='.2f')
svc=SVC()

model2=svc.fit(train_X,train_Y)

svc_predict = svc.predict(test_X)

acc_svc = accuracy_score(test_Y, svc_predict)

print(acc_svc)

print(classification_report(svc_predict,test_Y))

con  = confusion_matrix(svc_predict,test_Y)

sns.heatmap(con,annot=True,fmt='.2f')
gbc = GradientBoostingClassifier()

model3=gbc.fit(train_X,train_Y)

gbc_predict = gbc.predict(test_X)

acc_gbc = accuracy_score(test_Y, gbc_predict)

print(acc_gbc)

print(classification_report(gbc_predict,test_Y))

con  = confusion_matrix(gbc_predict,test_Y)

sns.heatmap(con,annot=True,fmt='.2f')

tree = DecisionTreeClassifier()

model5=tree.fit(train_X,train_Y)

tree_predict = tree.predict(test_X)

acc_tree = accuracy_score(test_Y, tree_predict)

print(acc_tree)

print(classification_report(tree_predict,test_Y))

con  = confusion_matrix(tree_predict,test_Y)

sns.heatmap(con,annot=True,fmt='.2f')

forest = RandomForestClassifier()

model6 = forest.fit(train_X,train_Y)

forest_predict = forest.predict(test_X)

acc_forest = accuracy_score(test_Y, forest_predict)

print(acc_forest)

print(classification_report(forest_predict,test_Y))

con  = confusion_matrix(forest_predict,test_Y)

sns.heatmap(con,annot=True,fmt='.2f')
knn=KNeighborsClassifier()

model7=knn.fit(train_X,train_Y)

knn_predict = knn.predict(test_X)

acc_knn = accuracy_score(test_Y, knn_predict)

print(acc_knn)

print(classification_report(knn_predict,test_Y))

con  = confusion_matrix(knn_predict,test_Y)

sns.heatmap(con,annot=True,fmt='.2f')
sgd =SGDClassifier()

model8=sgd.fit(train_X,train_Y)

sgd_predict = sgd.predict(test_X)

acc_sgd = accuracy_score(test_Y, sgd_predict)

print(acc_sgd)

print(classification_report(sgd_predict,test_Y))

con  = confusion_matrix(sgd_predict,test_Y)

sns.heatmap(con,annot=True,fmt='.2f')
print('Logistic Regression Accuracy:',round(acc_lg*100,2))

print('K-Nearest Neighbour Accuracy:',round(acc_knn*100,2))

print('Decision Tree Classifier Accuracy:',round(acc_tree*100,2))

print('Random Forest Classifier Accuracy:',round(acc_forest*100,2))

print('support Vector Machine Accuracy:',round(acc_svc*100,2))

print('GradientBoost Classifier Accuracy:',round(acc_gbc*100,2))

print('SGD Accuracy:',round(acc_sgd*100,2))
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=-1,train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training Examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="b")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



g = plot_learning_curve(model1," Logistic Regression learning curves",train_X,train_Y)

g = plot_learning_curve(model2," SVC learning curves",train_X,train_Y)

g = plot_learning_curve(model3," GradientBoost Classifier learning curves",train_X,train_Y)

g = plot_learning_curve(model8," SGD learning curves",train_X,train_Y)

g = plot_learning_curve(model5," Decision Tree Classifier learning curves",train_X,train_Y)

g = plot_learning_curve(model6," Random Forest Classifier learning curves",train_X,train_Y)

g = plot_learning_curve(model7," KNN learning curves",train_X,train_Y)
