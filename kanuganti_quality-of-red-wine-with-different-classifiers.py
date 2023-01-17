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
import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from matplotlib.ticker import FormatStrFormatter



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.tree import DecisionTreeClassifier ,export_graphviz 

import graphviz

from IPython.display import Image  # To plot decision tree.

from sklearn.externals.six import StringIO
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
df.info()
display(np.round(df.describe()))
df.groupby('quality').count()
null_columns=df.columns[df.isnull().any()]

print(null_columns)
# Finding the correlation bewteen the Features.

plt.figure(figsize=(10,5))

heatmap = sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap="Reds")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)

heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)

plt.show()
#relationship between the some of features

cols_sns = ['residual sugar', 'chlorides', 'density', 'pH', 'alcohol', 'quality']

sns.set(style="ticks")

sns.pairplot(df[cols_sns],hue="quality")
#BoxPlot for different features.

features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",

            "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]



fig = plt.figure(figsize=(16,8))

for i in range(len(features)):

    ax1 = fig.add_subplot(3, 4, i+1)

    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    sns.boxplot(x='quality', y=features[i], data=df,palette="Set3")

    i = i + 1

plt.subplots_adjust(hspace = 0.5)

plt.subplots_adjust(wspace = 0.5)

plt.show()
fig = plt.figure(figsize = (10,6))

sns.barplot(x='quality', y='sulphates',data=df)
from sklearn.preprocessing import LabelEncoder

bins = (2, 5.5, 8)

group_names = ['bad', 'good']

df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)



label_quality = LabelEncoder()

df['quality'] = label_quality.fit_transform(df['quality'])

df['quality'].value_counts()
X = df.drop('quality', axis=1)

y = df['quality']



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2)



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
print("X_train {0} , X_test {1} " .format(X_train.shape , X_test.shape))
#------------------------######### LogisticRegression #######-----------------

lrg_classifier = LogisticRegression(solver='newton-cg',tol= 0.0001,C= 0.5,)

lrg_classifier.fit(X_train, y_train.ravel())



cv_lr = cross_val_score(estimator = lrg_classifier, X = X_train, y = y_train.ravel(), cv = 10)

print("CV: ", cv_lr.mean())



lt_y_pred_train = lrg_classifier.predict(X_train)

accuracy_lr_train = accuracy_score(y_train, lt_y_pred_train)

print("Training set accuracy for Logistic Regression: ", accuracy_lr_train)



y_pred_lr_test = lrg_classifier.predict(X_test)

accuracy_lr_test = accuracy_score(y_test, y_pred_lr_test)

print("Test set accuracy for Logistic Regression: ", accuracy_lr_test)
lrg_classifier.fit(X_train, y_train)

lr_y_pred_test = lrg_classifier.predict(X_test)



print(classification_report(y_test, lr_y_pred_test))
confusion_matrix(y_test, lr_y_pred_test)



tp_lr = confusion_matrix(y_test, lr_y_pred_test)[0,0]

fp_lr = confusion_matrix(y_test, lr_y_pred_test)[0,1]

tn_lr = confusion_matrix(y_test, lr_y_pred_test)[1,1]

fn_lr = confusion_matrix(y_test, lr_y_pred_test)[1,0]
#------------------------#########  KNeighbors #######-----------------

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(leaf_size = 1, metric = 'minkowski', n_neighbors = 28, weights = 'distance')

knn_classifier.fit(X_train, y_train.ravel())
# Predicting Cross Validation Score

knn_cv = cross_val_score(estimator = knn_classifier, X = X_train, y = y_train.ravel(), cv = 10)

print("CV: ", knn_cv.mean())



knn_y_pred_train = knn_classifier.predict(X_train)

knn_accuracy_train = accuracy_score(y_train, knn_y_pred_train)

print("Training set accuracy for KNN: ", knn_accuracy_train)



knn_y_pred_test = knn_classifier.predict(X_test)

knn_accuracy_test = accuracy_score(y_test, knn_y_pred_test)

print("Test set accuracy for KNN: ", knn_accuracy_test)
confusion_matrix(y_test, knn_y_pred_test)



tp_knn = confusion_matrix(y_test, knn_y_pred_test)[0,0]

fp_knn = confusion_matrix(y_test, knn_y_pred_test)[0,1]

tn_knn = confusion_matrix(y_test, knn_y_pred_test)[1,1]

fn_knn = confusion_matrix(y_test, knn_y_pred_test)[1,0]
#------------------------#########  SVC #######-----------------

from sklearn.svm import SVC

svm_linear_classifier = SVC(kernel = 'linear')

svm_linear_classifier.fit(X_train, y_train.ravel())
svm_linear_cv = cross_val_score(estimator = svm_linear_classifier, X = X_train, y = y_train.ravel(), cv = 10)

print("CV: ", svm_linear_cv.mean())



svm_linear_train_y_pred = svm_linear_classifier.predict(X_train)



svm_linear_accuracy_train = accuracy_score(y_train, svm_linear_train_y_pred)

print("Training set accuracy for SVC: ", svm_linear_accuracy_train)



svm_linear_y_pred_test = svm_linear_classifier.predict(X_test)

svm_linear_accuracy_test = accuracy_score(y_test, svm_linear_y_pred_test)

print("Test set accuracy for SVC: ", svm_linear_accuracy_test)
confusion_matrix(y_test, svm_linear_y_pred_test)



tp_svm_linear = confusion_matrix(y_test, svm_linear_y_pred_test)[0,0]

fp_svm_linear = confusion_matrix(y_test, svm_linear_y_pred_test)[0,1]

tn_svm_linear = confusion_matrix(y_test, svm_linear_y_pred_test)[1,1]

fn_svm_linear = confusion_matrix(y_test, svm_linear_y_pred_test)[1,0]
#------------------------#########  DecisionTree Classifier #######-----------------



dt_classifier = DecisionTreeClassifier(criterion = 'gini', max_features=6, max_leaf_nodes=400, random_state = 33, max_depth=4)

dt_classifier.fit(X_train, y_train.ravel())
dt_cv = cross_val_score(estimator = dt_classifier, X = X_train, y = y_train.ravel(), cv = 10)

print("CV: ", dt_cv.mean())



dt_y_pred_train = dt_classifier.predict(X_train)

dt_accuracy_train = accuracy_score(y_train, dt_y_pred_train)

print("Training set accuracy for DecisionTree: ", dt_accuracy_train)



dt_y_pred_test = dt_classifier.predict(X_test)

dt_accuracy_test = accuracy_score(y_test, dt_y_pred_test)

print("Test set accuracy for DecisionTree: ", dt_accuracy_test)
confusion_matrix(y_test, dt_y_pred_test)



tp_dt = confusion_matrix(y_test, dt_y_pred_test)[0,0]

fp_dt = confusion_matrix(y_test, dt_y_pred_test)[0,1]

tn_dt = confusion_matrix(y_test, dt_y_pred_test)[1,1]

fn_dt = confusion_matrix(y_test, dt_y_pred_test)[1,0]
data = export_graphviz(dt_classifier,out_file=None,feature_names=list(X.columns.values),class_names=None,   

                         filled=True, rounded=True,  

                         special_characters=True,proportion=True)



graph = graphviz.Source(data)

graph
#------------------------#########  RandomForest Classifier #######-----------------

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(criterion = 'entropy', max_features = 4, n_estimators = 800, random_state=33)

rf_classifier.fit(X_train, y_train.ravel())
# Predicting Cross Validation Score

rf_cv = cross_val_score(estimator = rf_classifier, X = X_train, y = y_train.ravel(), cv = 10)

print("CV: ", rf_cv.mean())



rf_y_pred_train = rf_classifier.predict(X_train)

rf_accuracy_train = accuracy_score(y_train, rf_y_pred_train)

print("Training set: ", rf_accuracy_train)



rf_y_pred_test = rf_classifier.predict(X_test)

rf_accuracy_test = accuracy_score(y_test, rf_y_pred_test)

print("Test set: ", rf_accuracy_test)
#..--------Important Features of Random Forest

feature_importances = pd.DataFrame(rf_classifier.feature_importances_,

                                   index = X.columns,columns=['importance']).sort_values('importance', ascending=False)

print(feature_importances)
importance = pd.DataFrame({'Importance': rf_classifier.feature_importances_}, index=X.columns)

importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r')

plt.xlabel('Variable importance')

plt.legend(loc='lower right')

plt.legend()

plt.show()
confusion_matrix(y_test, rf_y_pred_test)



tp_rf = confusion_matrix(y_test, rf_y_pred_test)[0,0]

fp_rf = confusion_matrix(y_test, rf_y_pred_test)[0,1]

tn_rf = confusion_matrix(y_test, rf_y_pred_test)[1,1]

fn_rf = confusion_matrix(y_test, rf_y_pred_test)[1,0]
#------------

###-----------Comparsion between the different models of Accuracy and Cross Validation.



models = [('Logistic Regression', tp_lr, fp_lr, tn_lr, fn_lr, accuracy_lr_train, accuracy_lr_test, cv_lr.mean()),

          ('K-Nearest Neighbors (KNN)', tp_knn, fp_knn, tn_knn, fn_knn, knn_accuracy_train, knn_accuracy_test, knn_cv.mean()),

          ('SVM', tp_svm_linear, fp_svm_linear, tn_svm_linear, fn_svm_linear, svm_linear_accuracy_train, svm_linear_accuracy_test, svm_linear_cv.mean()),

          ('Decision Tree Classification', tp_dt, fp_dt, tn_dt, fn_dt, dt_accuracy_train, dt_accuracy_test, dt_cv.mean()),

          ('Random Forest Tree Classification', tp_rf, fp_rf, tn_rf, fn_rf, rf_accuracy_train, rf_accuracy_test, rf_cv.mean())

         ]



predict = pd.DataFrame(data = models, columns=['Model', 'True Positive', 'False Positive', 'True Negative','False Negative', 'Accuracy(training)', 'Accuracy(test)',

                                               'Cross-Validation'])

predict
f, axes = plt.subplots(2,1, figsize=(14,10))

predict.sort_values(by=['Accuracy(training)'], ascending=False, inplace=True)

sns.barplot(x='Accuracy(training)', y='Model', data = predict, palette='Blues_d', ax = axes[0])

#axes[0].set(xlabel='Region', ylabel='Charges')

axes[0].set_xlabel('Accuracy (Training)', size=16)

axes[0].set_ylabel('Model')

axes[0].set_xlim(0,1.0)

axes[0].set_xticks(np.arange(0, 1.1, 0.1))

predict.sort_values(by=['Accuracy(test)'], ascending=False, inplace=True)

sns.barplot(x='Accuracy(test)', y='Model', data = predict, palette='Reds_d', ax = axes[1])

#axes[0].set(xlabel='Region', ylabel='Charges')

axes[1].set_xlabel('Accuracy (Test)', size=16)

axes[1].set_ylabel('Model')

axes[1].set_xlim(0,1.0)

axes[1].set_xticks(np.arange(0, 1.1, 0.1))

plt.show()