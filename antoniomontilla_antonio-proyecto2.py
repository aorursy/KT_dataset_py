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
# analisis de datos

import pandas as pd

import numpy as np

import random as rnd

from pandas import read_csv



# visualización

import seaborn as sns

from scipy.stats import norm, skew

from scipy import stats

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import GradientBoostingClassifier



## scikit modeling libraries

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,

                             GradientBoostingClassifier, ExtraTreesClassifier,

                             VotingClassifier)



from sklearn.model_selection import (GridSearchCV, cross_val_score, cross_val_predict,

                                     StratifiedKFold, learning_curve)



## Predictive modeling

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve, auc

from sklearn.feature_selection import RFE



#Principal components & otros

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from sklearn.model_selection import (cross_val_score, 

                                     KFold, 

                                     train_test_split)

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
original = '../input/'

url1 = "../input/Train_new2.csv"

url2 = "../input/df_model.csv"



Train_new2 = pd.read_csv(url1)

df_model = read_csv(url2)
#Por eficiencia computacional, se procede a extraer una muestra de 100.000 observaciones para el modelado

df_sample = df_model.sample(frac =.305, random_state = 2)

print('El tamaño de la muestra de df_sample es: ', df_sample.shape)
# Dividir y eliminar variable objetivo de explicativas

X_train = df_sample.drop(['isFraud'], axis=1)

Y_train = df_sample['isFraud']

# Se crea train y test 80-20 con la semilla fijada en 42 para la validación de los modelos

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.20, random_state=42)

X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
# Creo un _performance_auc_ dict para la comparación de los modelos

performance_auc = {}
model = LogisticRegression().fit(X_train, Y_train)

predicted_log = model.predict(X_test)

predicted_log



# Confidence score

logreg_score_1 = round(model.score(X_train, Y_train) * 100, 2)



print(logreg_score_1)



print(classification_report(Y_test, predicted_log))
# Visualize results by ROC graph

fpr, tpr, thresholds = roc_curve(Y_test, predicted_log)

roc_auc = auc(fpr, tpr)

performance_auc['Logistic Regression'] = roc_auc



# Plotting

plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = DecisionTreeClassifier().fit(X_train, Y_train)

model
predicted_dt = model.predict(X_test)

predicted_dt
# Confidence score

dectree_score_1 = round(model.score(X_train, Y_train) * 100, 2)

print(dectree_score_1)

print(classification_report(Y_test, predicted_dt))
# Create a confusion matrix

matrix = confusion_matrix(Y_test, predicted_dt)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
# Visualize results by ROC graph

fpr, tpr, thresholds = roc_curve(Y_test, predicted_dt)

roc_auc = auc(fpr, tpr)

performance_auc['Decision Tree'] = roc_auc



# Plotting

plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_train, Y_train)

model
predicted_rf = model.predict(X_test)

predicted_rf
# Confidence score

randfor_score_1 = round(model.score(X_train,Y_train) * 100, 2)

print(randfor_score_1)

print(classification_report(Y_test, predicted_rf))
matrix = confusion_matrix(Y_test, predicted_rf)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
fpr, tpr, thresholds = roc_curve(Y_test, predicted_rf)

roc_auc = auc(fpr, tpr)

performance_auc['Random Forests'] = roc_auc



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = KNeighborsClassifier(3).fit(X_train, Y_train)

model
predicted_knn = model.predict(X_test)

predicted_knn
# Confidence score

knn_score_1 = round(model.score(X_train,Y_train) * 100, 2)

print(knn_score_1)

print(classification_report(Y_test, predicted_knn))
matrix = confusion_matrix(Y_test, predicted_knn)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
fpr, tpr, thresholds = roc_curve(Y_test, predicted_knn)

roc_auc = auc(fpr, tpr)

performance_auc['k-nearest neighbours'] = roc_auc



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = GaussianNB().fit(X_train, Y_train)

model
predicted_gau = model.predict(X_test)

predicted_gau
# Confidence score

gau_score_1 = round(model.score(X_train,Y_train) * 100, 2)

print(gau_score_1)

print(classification_report(Y_test, predicted_gau))
matrix = confusion_matrix(Y_test, predicted_gau)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
fpr, tpr, thresholds = roc_curve(Y_test, predicted_gau)

roc_auc = auc(fpr, tpr)

performance_auc['GaussianNB'] = roc_auc



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = GradientBoostingClassifier().fit(X_train, Y_train)

model
predicted_gbc = model.predict(X_test)

predicted_gbc
# Confidence score

gbc_score_1 = round(model.score(X_train,Y_train) * 100, 2)

print(gbc_score_1)

print(classification_report(Y_test, predicted_gbc))
matrix = confusion_matrix(Y_test, predicted_gbc)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
fpr, tpr, thresholds = roc_curve(Y_test, predicted_gbc)

roc_auc = auc(fpr, tpr)

performance_auc['GradientBoostingClassifier'] = roc_auc



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = XGBClassifier().fit(X_train, Y_train)

model
predicted_xgb = model.predict(X_test)

predicted_xgb
# Confidence score

xgb_score_1 = round(model.score(X_train,Y_train) * 100, 2)

print(xgb_score_1)

print(classification_report(Y_test, predicted_xgb))
matrix = confusion_matrix(Y_test, predicted_xgb)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
fpr, tpr, thresholds = roc_curve(Y_test, predicted_xgb)

roc_auc = auc(fpr, tpr)

performance_auc['XGBClassifier'] = roc_auc



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
perf = pd.DataFrame.from_dict(performance_auc, orient='index')

perf['Model'] = perf.index

perf['AUC'] = perf[0]

plt.xlabel('AUC')

plt.title('Classifier AUC')

sns.set_color_codes("muted")

sns.barplot(x='AUC', y='Model', data=perf, color="b")
#Características de la variable objetivo _isFraud_

print('El tamaño de df_model es: ', df_model.shape)

print("La variable objetivo _Is Fraud_ tiene {0} obervaciones y {1} son valores únicos.".format(df_model['isFraud'].count(),df_model['isFraud'].nunique()))

print(df_model['isFraud'].value_counts())
#1) Creando la nueva dataframe

#Extrayendo la nueva muestra con valores _IsFraud_ = 1

df_model_fraud = df_model.loc[df_model['isFraud'] == 1]

print('El tamaño de df_model_fraud es: ', df_model_fraud.shape)

print(df_model_fraud['isFraud'].value_counts())

#Ahora lo equivalente con valores _isFraud_ = 0

df_model_nofraud = df_model.loc[df_model['isFraud'] == 0]

print('El tamaño de df_model_nofraud es: ', df_model_nofraud.shape)

print(df_model_nofraud['isFraud'].value_counts())

#Muestra de _isFraud_ de tamaño 6506

df_sample_nofraud = df_model_nofraud.sample(frac =.02022431, random_state = 2)

print('El tamaño de df_sample_nofraud es: ', df_sample_nofraud.shape)

#Concatenando las dos samples

df_sample2 = pd.concat([df_model_fraud, df_sample_nofraud])

print('El tamaño de df_sample2 es: ', df_sample2.shape)

print(df_sample2['isFraud'].value_counts())
#2) Separando la nueva df en train y test

# Dividir y eliminar variable objetivo de explicativas

X_train = df_sample2.drop(['isFraud'], axis=1)

Y_train = df_sample2['isFraud']

# Se crea train ay test 80-20 con la semilla fijada en 42 para la validación de los modelos

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.20, random_state=42)

X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

performance_auc2 = {}
model = LogisticRegression().fit(X_train, Y_train)

predicted_log = model.predict(X_test)

predicted_log



# Confidence score

logreg_score_1 = round(model.score(X_train, Y_train) * 100, 2)



print(logreg_score_1)



print(classification_report(Y_test, predicted_log))
# Create a confusion matrix

matrix = confusion_matrix(Y_test, predicted_log)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
# Visualize results by ROC graph

fpr, tpr, thresholds = roc_curve(Y_test, predicted_log)

roc_auc = auc(fpr, tpr)

performance_auc2['Logistic Regression'] = roc_auc



# Plotting

plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = DecisionTreeClassifier().fit(X_train, Y_train)

model
predicted_dt = model.predict(X_test)

predicted_dt
# Confidence score

dectree_score_1 = round(model.score(X_train, Y_train) * 100, 2)

print(dectree_score_1)

print(classification_report(Y_test, predicted_dt))
# Create a confusion matrix

matrix = confusion_matrix(Y_test, predicted_dt)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
# Visualize results by ROC graph

fpr, tpr, thresholds = roc_curve(Y_test, predicted_dt)

roc_auc = auc(fpr, tpr)

performance_auc2['Decision Tree'] = roc_auc



# Plotting

plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_train, Y_train)

model
predicted_rf = model.predict(X_test)

predicted_rf
# Confidence score

randfor_score_1 = round(model.score(X_train,Y_train) * 100, 2)

print(randfor_score_1)

print(classification_report(Y_test, predicted_rf))
matrix = confusion_matrix(Y_test, predicted_rf)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
fpr, tpr, thresholds = roc_curve(Y_test, predicted_rf)

roc_auc = auc(fpr, tpr)

performance_auc2['Random Forests'] = roc_auc



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = KNeighborsClassifier(3).fit(X_train, Y_train)

model
predicted_knn = model.predict(X_test)

predicted_knn
# Confidence score

knn_score_1 = round(model.score(X_train,Y_train) * 100, 2)

print(knn_score_1)

print(classification_report(Y_test, predicted_knn))
matrix = confusion_matrix(Y_test, predicted_knn)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
fpr, tpr, thresholds = roc_curve(Y_test, predicted_knn)

roc_auc = auc(fpr, tpr)

performance_auc2['k-nearest neighbours'] = roc_auc



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = GaussianNB().fit(X_train, Y_train)

model
predicted_gau = model.predict(X_test)

predicted_gau
# Confidence score

gau_score_1 = round(model.score(X_train,Y_train) * 100, 2)

print(gau_score_1)

print(classification_report(Y_test, predicted_gau))
matrix = confusion_matrix(Y_test, predicted_gau)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
fpr, tpr, thresholds = roc_curve(Y_test, predicted_gau)

roc_auc = auc(fpr, tpr)

performance_auc2['GaussianNB'] = roc_auc



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = GradientBoostingClassifier().fit(X_train, Y_train)

model
predicted_gbc = model.predict(X_test)

predicted_gbc
# Confidence score

gbc_score_1 = round(model.score(X_train,Y_train) * 100, 2)

print(gbc_score_1)

print(classification_report(Y_test, predicted_gbc))
matrix = confusion_matrix(Y_test, predicted_gbc)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
fpr, tpr, thresholds = roc_curve(Y_test, predicted_gbc)

roc_auc = auc(fpr, tpr)

performance_auc2['GradientBoostingClassifier'] = roc_auc



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = XGBClassifier().fit(X_train, Y_train)

model
predicted_xgb = model.predict(X_test)

predicted_xgb
# Confidence score

xgb_score_1 = round(model.score(X_train,Y_train) * 100, 2)

print(xgb_score_1)

print(classification_report(Y_test, predicted_xgb))
matrix = confusion_matrix(Y_test, predicted_xgb)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
fpr, tpr, thresholds = roc_curve(Y_test, predicted_xgb)

roc_auc = auc(fpr, tpr)

performance_auc2['XGBClassifier'] = roc_auc



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
perf = pd.DataFrame.from_dict(performance_auc2, orient='index')

perf['Model'] = perf.index

perf['AUC'] = perf[0]

plt.xlabel('AUC')

plt.title('Classifier AUC')

sns.set_color_codes("muted")

sns.barplot(x='AUC', y='Model', data=perf, color="b")
#1) Creando la nueva dataframe

#Extrayendo la nueva muestra con valores _IsFraud_ = 1

df_model_fraud = df_model.loc[df_model['isFraud'] == 1]

print('El tamaño de df_model_fraud es: ', df_model_fraud.shape)

print(df_model_fraud['isFraud'].value_counts())

#Ahora lo equivalente con valores _isFraud_ = 0

df_model_nofraud = df_model.loc[df_model['isFraud'] == 0]

print('El tamaño de df_model_nofraud es: ', df_model_nofraud.shape)

print(df_model_nofraud['isFraud'].value_counts())

#Muestra de _isFraud_ de tamaño 6506

df_sample_nofraud = df_model_nofraud.sample(frac =.04718799, random_state = 2)

print('El tamaño de df_sample_nofraud es: ', df_sample_nofraud.shape)

#Concatenando las dos samples

df_sample2 = pd.concat([df_model_fraud, df_sample_nofraud])

print('El tamaño de df_sample2 es: ', df_sample2.shape)

print(df_sample2['isFraud'].value_counts())
#2) Separando la nueva df en train y test

# Dividir y eliminar variable objetivo de explicativas

X_train = df_sample2.drop(['isFraud'], axis=1)

Y_train = df_sample2['isFraud']

# Se crea train ay test 80-20 con la semilla fijada en 42 para la validación de los modelos

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.20, random_state=42)

X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

performance_auc3 = {}
model = LogisticRegression().fit(X_train, Y_train)

predicted_log = model.predict(X_test)

predicted_log



# Confidence score

logreg_score_1 = round(model.score(X_train, Y_train) * 100, 2)



print(logreg_score_1)



print(classification_report(Y_test, predicted_log))
# Create a confusion matrix

matrix = confusion_matrix(Y_test, predicted_log)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
# Visualize results by ROC graph

fpr, tpr, thresholds = roc_curve(Y_test, predicted_log)

roc_auc = auc(fpr, tpr)

performance_auc3['Logistic Regression'] = roc_auc



# Plotting

plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_train, Y_train)

model
predicted_rf = model.predict(X_test)

predicted_rf
# Confidence score

randfor_score_1 = round(model.score(X_train,Y_train) * 100, 2)

print(randfor_score_1)

print(classification_report(Y_test, predicted_rf))
matrix = confusion_matrix(Y_test, predicted_rf)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
fpr, tpr, thresholds = roc_curve(Y_test, predicted_rf)

roc_auc = auc(fpr, tpr)

performance_auc3['Random Forests'] = roc_auc



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = XGBClassifier().fit(X_train, Y_train)

model
predicted_xgb = model.predict(X_test)

predicted_xgb
# Confidence score

xgb_score_1 = round(model.score(X_train,Y_train) * 100, 2)

print(xgb_score_1)

print(classification_report(Y_test, predicted_xgb))
matrix = confusion_matrix(Y_test, predicted_xgb)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
fpr, tpr, thresholds = roc_curve(Y_test, predicted_xgb)

roc_auc = auc(fpr, tpr)

performance_auc3['XGBClassifier'] = roc_auc



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
#Comparativa 

perf = pd.DataFrame.from_dict(performance_auc3, orient='index')

perf['Model'] = perf.index

perf['AUC'] = perf[0]

plt.xlabel('AUC')

plt.title('Classifier AUC')

sns.set_color_codes("muted")

sns.barplot(x='AUC', y='Model', data=perf, color="b")