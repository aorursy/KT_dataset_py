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



#시각화를 위한 그래프 라이브러리

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

#그래프 함수

def bar_chart(feature):

    yes = heart[heart['target']==1][feature].value_counts()

    no = heart[heart['target']==0][feature].value_counts()

    df = pd.DataFrame([yes,no])

    df.index = ['Yes','No']

    df.plot(kind='bar',stacked=True, figsize=(10,5))



# Any results you write to the current directory are saved as output.



heart = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

heart.head()



from sklearn.model_selection import train_test_split

#heart, heart2 = train_test_split(heart, test_size = 0.3, random_state = 123)

heart.info()
a = pd.get_dummies(heart['cp'], prefix = "cp")

b = pd.get_dummies(heart['thal'], prefix = "thal")

c = pd.get_dummies(heart['slope'], prefix = "slope")

d = pd.get_dummies(heart['restecg'], prefix = "restecg")

frames = [a, b, c, d, heart]

heart = pd.concat(frames, axis = 1)

heart = heart.drop(columns = ['cp', 'thal', 'slope', 'restecg'])



from sklearn import preprocessing 

x = heart.values.astype(float)

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

heart = pd.DataFrame(x_scaled, columns = heart.columns)

heart.shape

heart_train, heart_test = train_test_split(heart, test_size = 0.2, random_state = 0)
heart.info()



#age: 나이

#sex: 성별
heart.isnull().sum()
bar_chart('sex')
bar_chart('fbs')
bar_chart('exang')
bar_chart('ca')
import matplotlib.pyplot as plt

import seaborn as sns



facet = sns.FacetGrid(heart, hue="target", aspect=4)

facet.map(sns.kdeplot, 'age', shade=True)

facet.set(xlim=(0, heart['age'].max()))

facet.add_legend()

sns.axes_style("darkgrid")



plt.show()
facet = sns.FacetGrid(heart, hue="target", aspect=4)

facet.map(sns.kdeplot, 'trestbps', shade=True)

facet.set(xlim=(0, heart['trestbps'].max()))

facet.add_legend()

sns.axes_style("darkgrid")



plt.show()
facet = sns.FacetGrid(heart, hue="target", aspect=4)

facet.map(sns.kdeplot, 'chol', shade=True)

facet.set(xlim=(0, heart['chol'].max()))

facet.add_legend()

sns.axes_style("darkgrid")



plt.show()
facet = sns.FacetGrid(heart, hue="target", aspect=4)

facet.map(sns.kdeplot, 'thalach', shade=True)

facet.set(xlim=(0, heart['thalach'].max()))

facet.add_legend()

sns.axes_style("darkgrid")



plt.show()
facet = sns.FacetGrid(heart, hue="target", aspect=4)

facet.map(sns.kdeplot, 'oldpeak', shade=True)

facet.set(xlim=(0, heart['oldpeak'].max()))

facet.add_legend()

sns.axes_style("darkgrid")



plt.show()
heart.corr()
plt.figure(figsize=(20,20))

sns.heatmap(heart.corr(),vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

plt.tight_layout()

plt.show()


from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier


x_heart = heart_train.drop(['target'], axis=1)

y_heart = heart_train['target']

x2_heart = heart_test.drop(['target'], axis=1)

y2_heart = heart_test['target']

from sklearn.model_selection import cross_val_score
l1r = LogisticRegression(penalty = 'l1', C=0.65)

l1r.fit(x_heart, y = y_heart)

score = cross_val_score(l1r, heart.drop(['target'], axis=1), heart['target'], cv=5)

print("testset accuracy:",l1r.score(x2_heart, y2_heart))

print("cross validation accuracy:",score.mean())
l2r = LogisticRegression(penalty = 'l2', C=1)

l2r.fit(x_heart, y = y_heart)

score = cross_val_score(l2r, heart.drop(['target'], axis=1), heart['target'], cv=5)

print("testset accuracy:",l2r.score(x2_heart, y2_heart))

print("cross validation accuracy:",score.mean())
svc = SVC(C=0.35)

score = cross_val_score(svc, heart.drop(['target'], axis=1), heart['target'], cv=5)

svc.fit(x_heart, y_heart)

print("testset accuracy:",svc.score(x2_heart, y2_heart))

print("cross validation accuracy:",score.mean())
lsvc = SVC(kernel='linear', C=0.8)

lsvc.fit(x_heart, y = y_heart)

score = cross_val_score(lsvc, heart.drop(['target'], axis=1), heart['target'], cv=5)

print("testset accuracy:",lsvc.score(x2_heart, y2_heart))

print("cross validation accuracy:",score.mean())

dt = DecisionTreeClassifier()

dt.fit(x_heart, y = y_heart)

score = cross_val_score(dt, heart.drop(['target'], axis=1), heart['target'], cv=5)

print("testset accuracy:",dt.score(x2_heart, y2_heart))

print("cross validation accuracy:",score.mean())
rf = RandomForestClassifier(n_estimators = 1000)

rf.fit(x_heart, y = y_heart)

score = cross_val_score(rf, heart.drop(['target'],axis=1), heart['target'], cv=5)

print("testset accuracy:",rf.score(x2_heart, y2_heart))

print("cross validation accuracy:",score.mean())
y_head_l1r = l1r.predict(x2_heart)

y_head_l2r = l2r.predict(x2_heart)

y_head_svc = svc.predict(x2_heart)

y_head_lsvc = lsvc.predict(x2_heart)

y_head_dt = dt.predict(x2_heart)

y_head_rf = rf.predict(x2_heart)

from sklearn.metrics import confusion_matrix

cm_l1r = confusion_matrix(y2_heart,y_head_l1r)

cm_l2r = confusion_matrix(y2_heart,y_head_l2r)

cm_svc = confusion_matrix(y2_heart,y_head_svc)

cm_lsvc = confusion_matrix(y2_heart,y_head_lsvc)

cm_dt = confusion_matrix(y2_heart,y_head_dt)

cm_rf = confusion_matrix(y2_heart,y_head_rf)



plt.figure(figsize=(24,12))



plt.suptitle("Confusion Matrixes",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



plt.subplot(2,3,1)

plt.title("Logistic Regression(L1) Confusion Matrix")

sns.heatmap(cm_l1r,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,2)

plt.title("Logistic Regression(L2) Confusion Matrix")

sns.heatmap(cm_l2r,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,3)

plt.title("Support Vector Machine Confusion Matrix")

sns.heatmap(cm_svc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,4)

plt.title("Support Vector Machine(Linear) Confusion Matrix")

sns.heatmap(cm_lsvc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,5)

plt.title("Decision Tree Classifier Confusion Matrix")

sns.heatmap(cm_dt,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,6)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.show()
from sklearn.metrics import roc_curve, auc #for model evaluation

total=sum(sum(cm_l1r))



sensitivity = cm_l1r[0,0]/(cm_l1r[0,0]+cm_l1r[1,0])

print('Sensitivity : ', sensitivity )



specificity = cm_l1r[1,1]/(cm_l1r[1,1]+cm_l1r[0,1])

print('Specificity : ', specificity)

fpr, tpr, thresholds = roc_curve(y2_heart, y_head_l1r)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for logistic l1 regression classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)

auc(fpr, tpr)
total=sum(sum(cm_l2r))



sensitivity = cm_l2r[0,0]/(cm_l2r[0,0]+cm_l2r[1,0])

print('Sensitivity : ', sensitivity )



specificity = cm_l2r[1,1]/(cm_l2r[1,1]+cm_l2r[0,1])

print('Specificity : ', specificity)

fpr, tpr, thresholds = roc_curve(y2_heart, y_head_l2r)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for logistic l2 regression classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)

auc(fpr, tpr)
total=sum(sum(cm_svc))



sensitivity = cm_svc[0,0]/(cm_svc[0,0]+cm_svc[1,0])

print('Sensitivity : ', sensitivity )



specificity = cm_svc[1,1]/(cm_svc[1,1]+cm_svc[0,1])

print('Specificity : ', specificity)

fpr, tpr, thresholds = roc_curve(y2_heart, y_head_svc)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for svc classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)

auc(fpr, tpr)
total=sum(sum(cm_lsvc))



sensitivity = cm_lsvc[0,0]/(cm_lsvc[0,0]+cm_lsvc[1,0])

print('Sensitivity : ', sensitivity )



specificity = cm_lsvc[1,1]/(cm_lsvc[1,1]+cm_lsvc[0,1])

print('Specificity : ', specificity)

fpr, tpr, thresholds = roc_curve(y2_heart, y_head_lsvc)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for svc(linear) classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)

auc(fpr, tpr)
total=sum(sum(cm_dt))



sensitivity = cm_dt[0,0]/(cm_dt[0,0]+cm_dt[1,0])

print('Sensitivity : ', sensitivity )



specificity = cm_dt[1,1]/(cm_dt[1,1]+cm_dt[0,1])

print('Specificity : ', specificity)

fpr, tpr, thresholds = roc_curve(y2_heart, y_head_dt)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for decision tree classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)

auc(fpr, tpr)
total=sum(sum(cm_rf))



sensitivity = cm_rf[0,0]/(cm_rf[0,0]+cm_rf[1,0])

print('Sensitivity : ', sensitivity )



specificity = cm_rf[1,1]/(cm_rf[1,1]+cm_rf[0,1])

print('Specificity : ', specificity)

fpr, tpr, thresholds = roc_curve(y2_heart, y_head_rf)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for random forest classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)

auc(fpr, tpr)
import shap







explainer = shap.TreeExplainer(rf)

shap_values = explainer.shap_values(x_heart)

shap.initjs()

#shap.force_plot(explainer.expected_value[1], shap_values[1], x_heart)





plt.figure(figsize=(24,12))



plt.suptitle("shap",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



plt.subplot(2,3,1)

plt.title("random forest shap")

shap.summary_plot(shap_values[1], x_heart, plot_type="bar")

explainer = shap.TreeExplainer(dt)

shap_values = explainer.shap_values(x_heart)

plt.subplot(2,3,2)

plt.title("decision tree shap")

shap.summary_plot(shap_values[1], x_heart, plot_type="bar")



explainer = shap.KernelExplainer(l1r.predict_proba, x_heart)

shap_values = explainer.shap_values(x_heart)

plt.subplot(2,3,3)

plt.title("Logistic Regression(L1) shap")

shap.summary_plot(shap_values[1], x_heart, plot_type="bar")

plt.show()