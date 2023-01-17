# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/heart.csv")

df.head()
df.info()

df.describe()
import seaborn as sns

import matplotlib.pyplot as plt

def correlation_heatmap(dataframe):

    correlations = dataframe.corr()

    sns.heatmap(correlations, vmax=1.0, center=0, fmt=".2f", square=True, 

               linewidth=.5, annot=True, cmap="coolwarm", 

                cbar_kws={"shrink":.70})

    fig=plt.gcf()

    fig.set_size_inches(10,8)

    plt.show()

correlation_heatmap(df)

sns.pairplot(df)
df.isnull().sum().sum()
X = df.drop(["target"],1)

y = df["target"]
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.33,

                                                   random_state=2)
train_x.shape, test_x.shape, train_y.shape, test_y.shape
from tpot import TPOTClassifier

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2,

                     n_jobs=-1)
tpot.fit(train_x, train_y)
print("Accuracy is {}".format(tpot.score(test_x,test_y)))


from sklearn.model_selection import StratifiedKFold,cross_val_score,cross_val_predict

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,classification_report
k_fold = StratifiedKFold(n_splits=10)

classifiers = []
classifiers.append(LogisticRegression())

classifiers.append(KNeighborsClassifier())

classifiers.append(DecisionTreeClassifier())

classifiers.append(svm.SVC())

classifiers.append(RandomForestClassifier())

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(),learning_rate=0.1))

classifiers.append(GradientBoostingClassifier())

classifiers.append(ExtraTreesClassifier())
results_list=[]

means=[]

stds=[]



for classifier in classifiers:

    results_list.append(cross_val_score(classifier,X,y, scoring='accuracy',

                                      cv=k_fold,n_jobs=-1))



for i in results_list:

    means.append(i.mean())

    stds.append(i.std())

    

cv_res=pd.DataFrame({'cross_val_means':means,'cross_val_errors':stds,

                     'Algorithm':['Logistic Regression','Decision Tree',

                                  'Random Forest','AdaBoost',

                                  'Gradient Boosting',

                                  'Extra Trees Classifier','SVM', 'KNN']})    

g=sns.barplot(cv_res.cross_val_means,cv_res.Algorithm,data=cv_res,

              palette='coolwarm',orient='h',**{'xerr':stds})

g.set_xlabel('Mean Accuracy')

g=g.set_title('Cross Validation Scores')
logreg = LogisticRegression()

logreg.fit(train_x, train_y)

y_hat = logreg.predict(test_x)

print("Accuracy is {}".format(accuracy_score(test_y, y_hat)))
knn = KNeighborsClassifier(n_neighbors=20)

knn.fit(train_x, train_y)

y_hat_knn = knn.predict(test_x)

print("Accuracy is {}".format(accuracy_score(test_y, y_hat_knn)))

Ks = 20

mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))

ConfustionMx = [];

for n in range(1,Ks):

    

    #Train Model and Predict  

    neigh = KNeighborsClassifier(n_neighbors = n).fit(train_x,train_y)

    yhat=neigh.predict(test_x)

    mean_acc[n-1] = accuracy_score(test_y, yhat)



    

    std_acc[n-1]=np.std(yhat==test_y)/np.sqrt(yhat.shape[0])

mean_acc
plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)

plt.legend(('Accuracy ', '+/- 3xstd'))

plt.ylabel('Accuracy ')

plt.xlabel('Number of Nabors (K)')

plt.tight_layout()

plt.show()



print( "The best accuracy was with", mean_acc.max(), "with k=", 

      mean_acc.argmax()+1) 

confusion_m=confusion_matrix(test_y,y_hat)
sns.heatmap(confusion_m,annot=True,fmt='d',cmap='coolwarm')

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.xlabel('True')

plt.ylabel('Predicted')

plt.title('Confusion Matrix for SVM')

plt.show();
print(classification_report(test_y,y_hat))
from sklearn.pipeline import make_pipeline, make_union

from tpot.builtins import StackingEstimator



features = X.values

training_x,testing_x,training_y,testing_y=train_test_split(features, y.values, random_state=None)



exported_pipeline = make_pipeline(

    StackingEstimator(estimator=LogisticRegression(C=0.5,solver="liblinear",dual=False,penalty="l2")),

    LogisticRegression(C=20.0, dual=False, penalty="l2"))



exported_pipeline.fit(training_x, training_y)

results = exported_pipeline.predict(testing_x)

results