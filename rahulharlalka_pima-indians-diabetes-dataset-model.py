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
dataset=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

dataset.head()
import seaborn as sns

import matplotlib.pyplot as plt

print("no of people with no diabestes ",dataset.Outcome.value_counts()[0])

print("no of people with diabestes ",dataset.Outcome.value_counts()[1])

sns.countplot(dataset.Outcome)

plt.savefig('no of values of the each datapoint.jpg')
heat=dataset.corr()

sns.heatmap(heat)

plt.savefig('correlation_heatmap.jpg')
sns.pairplot(dataset,hue='Outcome')

plt.savefig('fig3.jpg')
sns.regplot(x=dataset.Pregnancies,y=dataset.Outcome)

plt.savefig('pregnancy vs outcome relation.jpg')
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif
x=dataset.drop(['Outcome'],axis=1)

x.head()

x=np.array(x)

print(x.shape)
y=dataset[['Outcome']]

y.head()

y=np.array(y).ravel()

print(y.shape)
test=SelectKBest(score_func=f_classif,k='all')

fit=test.fit(x,y)

print("the signicance of the respective data features wrt to the output variable are \n",fit.scores_)
test1=SelectKBest(score_func=f_classif,k=5)

fit1=test1.fit(x,y)

print(fit1.scores_)
x=fit1.transform(x)

sns.distplot(x[:,0],rug=True,kde=True)

plt.savefig('distribution of the preganancies throughout the dataset.jpg')
sns.distplot(x[:,1],rug=True,kde=True)

plt.savefig('glusose distribution.jpg')
sns.distplot(x[:,2],rug=True,kde=True)

plt.savefig('BMI distribution.jpg')
g=sns.PairGrid(dataset)

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels=6)

plt.savefig('fig7.jpg')
sns.boxplot(x='Pregnancies',y='Glucose',hue='Outcome',data=dataset)

plt.savefig('fig8.jpg')
sns.clustermap(heat)

plt.savefig('fig9.jpg')
X_data=pd.DataFrame(x,columns=['Pregnancies','Glucose','BMI','diabetespedigreefunction','age'])

X_data.head()
Y_data=pd.DataFrame(y,columns=['Outcome'])

Y_data.head()
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X_data,Y_data,test_size=0.25)
sns.countplot(np.array(Y_train).ravel())

plt.savefig('count of datapoints in y_train.jpg')
from imblearn.over_sampling import SMOTE 

sm = SMOTE(random_state = 2) 

X_train,Y_train=sm.fit_sample(np.array(X_train),np.array(Y_train).ravel())
sns.countplot(Y_train.ravel())

plt.savefig('count of datapoint in the train data.jpg')
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve,confusion_matrix,roc_auc_score,accuracy_score
model=LogisticRegression(solver='lbfgs',max_iter=200)

model.fit(X_train,Y_train)
Y_predict_probs=model.predict_proba(X_test)

Y_predict_probs=Y_predict_probs[:,1]
lr_auc=roc_auc_score(Y_test,Y_predict_probs)

print("the area under the roc curve is ",lr_auc)
fpr,tpr,thresholds=roc_curve(Y_test,Y_predict_probs)
plt.plot([0,1], [0,1], linestyle='--', label='No Skill')

plt.plot(fpr, tpr, marker='.', label='Logistic')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.show()

plt.savefig('roc_curve.jpg')
g_means= np.sqrt(tpr*(1-fpr))

ix=np.argmax(g_means)

print("the optimum threshold value for the given dataset is ", thresholds[ix])
plt.plot([0,1], [0,1], linestyle='--', label='No Skill')

plt.plot(fpr, tpr, marker='.', label='Logistic')

plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.show()

plt.savefig('roc_curve_optimum_value.jpg')
THRESHOLD=thresholds[ix]

y_predict=np.where(model.predict_proba(X_test)[:,1]>THRESHOLD ,1,0)

acc=accuracy_score(Y_test,y_predict)

print("the accuracy of the model is ",acc)

confusion_matrix(Y_test,y_predict)
model2=LogisticRegression()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

model2.fit(x_train,y_train)

preds=model2.predict(x_test)

print("the accuracy of the model without feature engineering is ",accuracy_score(y_test,preds))
confusion_matrix(y_test,preds)