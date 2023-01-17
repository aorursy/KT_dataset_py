# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random 



random.seed(19)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load data



df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
pd.set_option('display.max_columns', 110)
df.head()
df.describe()
df.isna().sum()[1:30]
#original_columns = df.columns.values



#for i in range(len(df.index)) :

#    print("Nan in row ", i , " : " ,  df[original_columns[6:110]].iloc[i].isnull().sum())
df.groupby("SARS-Cov-2 exam result").count()
df.groupby("SARS-Cov-2 exam result").count()
set_columns = ['Patient ID', 'Patient age quantile', 'SARS-Cov-2 exam result',

       'Patient addmited to regular ward (1=yes, 0=no)',

       'Patient addmited to semi-intensive unit (1=yes, 0=no)',

       'Patient addmited to intensive care unit (1=yes, 0=no)',

       'Hematocrit', 'Hemoglobin', 'Platelets', 'Mean platelet volume ',

       'Red blood Cells', 'Lymphocytes','Mean corpuscular hemoglobin concentration\xa0(MCHC)',

       'Leukocytes', 'Basophils', 'Mean corpuscular hemoglobin (MCH)',

       'Eosinophils', 'Mean corpuscular volume (MCV)', 'Monocytes',

       'Red blood cell distribution width (RDW)','Respiratory Syncytial Virus','Influenza A',

       'Influenza B','Parainfluenza 1','CoronavirusNL63','Rhinovirus/Enterovirus',

       'Coronavirus HKU1', 'Parainfluenza 3','Chlamydophila pneumoniae', 'Adenovirus', 'Parainfluenza 4',

       'Coronavirus229E', 'CoronavirusOC43', 'Inf A H1N1 2009',

       'Bordetella pertussis', 'Metapneumovirus', 'Parainfluenza 2']
df = df[set_columns]

df.head()
df.columns = [x.lower().strip().replace(' ','_') for x in df.columns]

df.columns
df['sars-cov-2_exam_result'] = [1 if a == 'positive' else 0 for a in df['sars-cov-2_exam_result'].values]



for i in df.columns[20:]:

    df[i] = [1 if a == 'detected' else 0 for a in df[i].values]

    

df.head(20)
df.describe()
df.isnull().sum()
df = df.dropna()
df.describe()
df.groupby("sars-cov-2_exam_result").count()
df = df[df.columns[:20]]
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



corrmat = df.corr()

f, ax = plt.subplots(figsize=(16, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
corrmat_v1 = corrmat.nlargest(10, 'sars-cov-2_exam_result')



features = corrmat_v1.index.values.tolist()



sns.heatmap(df[features].corr(), yticklabels=features, xticklabels=features, square=True);
# features



print('Features: ',features[1:])

print('Target: ',features[0])
from sklearn.model_selection import train_test_split



X = df[features[1:]]

Y = df[features[0]]





X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=7)



print('Number of training observations: ',len(X_train))

print('Number of test observations: ',len(X_test))
# Metrics



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



def plot_confusion_matrix(cm):

    ax= plt.subplot()

    sns.heatmap(cm, annot=True, ax = ax,fmt='g',cmap='Blues');

    ax.set_xlabel('Predict');ax.set_ylabel('True'); 

    ax.set_title('Confusion matrix'); 

    ax.xaxis.set_ticklabels(['Negative', 'Positive']); ax.yaxis.set_ticklabels(['Negative','Positive']);
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



y_pred = logreg.predict(X_test)



predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(Y_test, predictions)



print("Accuracy: %.2f%%" % (accuracy * 100.0))
cf_matrix = confusion_matrix(predictions,Y_test)



plot_confusion_matrix(cf_matrix)
print(classification_report(Y_test, predictions))
from sklearn import tree



model_tree = tree.DecisionTreeClassifier()

model_tree = model_tree.fit(X_train, Y_train)



y_pred = model_tree.predict(X_test)



predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(Y_test, predictions)



print("Accuracy: %.2f%%" % (accuracy * 100.0))
fig, ax = plt.subplots(figsize=(15,15))

tm = tree.plot_tree(model_tree, ax=ax)

plt.show()
cf_matrix = confusion_matrix(predictions,Y_test)



plot_confusion_matrix(cf_matrix)
print(classification_report(Y_test, predictions))
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict





tree_cls = tree.DecisionTreeClassifier()

scores = cross_val_score(tree_cls, X, Y,

                        scoring='neg_mean_squared_error', cv=10)



y_pred = cross_val_predict(tree_cls, X, Y, cv=10)



predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(Y, predictions)



print("Accuracy: %.2f%%" % (accuracy * 100.0))
cf_matrix = confusion_matrix(predictions,Y)



plot_confusion_matrix(cf_matrix)
print(classification_report(Y, predictions))
from sklearn.ensemble import RandomForestClassifier



model_rf = RandomForestClassifier(max_depth=2, random_state=7)

model_rf = model_rf.fit(X_train, Y_train)



y_pred = model_rf.predict(X_test)



predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(Y_test, predictions)



print("Accuracy: %.2f%%" % (accuracy * 100.0))
cf_matrix = confusion_matrix(predictions,Y_test)



plot_confusion_matrix(cf_matrix)
#print(classification_report(Y_test, predictions))
from xgboost import XGBClassifier



model_xgb = XGBClassifier()



model_xgb.fit(X_train, Y_train)



y_pred = model_xgb.predict(X_test)



predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(Y_test, predictions)



print("Accuracy: %.2f%%" % (accuracy * 100.0))
cf_matrix = confusion_matrix(predictions,Y_test)



plot_confusion_matrix(cf_matrix)
print(classification_report(Y_test, predictions))