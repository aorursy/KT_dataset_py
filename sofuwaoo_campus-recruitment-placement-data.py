import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



#from pandas_profiling import ProfileReport

#from autoviz.AutoViz_Class import AutoViz_Class as AVC



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score
raw_data = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

raw_data.head()
raw_data.describe(include='all')
raw_data.isnull().sum()
raw_data['status'].value_counts()
raw_data['salary'] = raw_data['salary'].fillna(0)

raw_data['salary'].isnull().sum()
raw_data['salary'].skew()
sns.boxplot(raw_data['salary'])
fig = plt.figure(figsize=(8,6))

sns.countplot(data= raw_data, x = 'status')

plt.xlabel('Status', fontsize = 14)

plt.ylabel('Count', fontsize = 14)

plt.title('Placement Rate', fontsize = 18)

plt.show()
fig = plt.figure(figsize=(8,6))

sns.countplot(data= raw_data, hue = 'workex', x = 'status')

plt.xlabel('Status', fontsize = 14)

plt.ylabel('Count', fontsize = 14)

plt.title('Placement Rate based on Work Experience', fontsize = 18)

plt.show()
fig = plt.figure(figsize=(8,6))

sns.countplot(data= raw_data, hue = 'gender', x = 'status')

plt.xlabel('Status', fontsize = 14)

plt.ylabel('Count', fontsize = 14)

plt.title('Placement Rate by Gender', fontsize = 18)

plt.show()
fig = plt.figure(figsize=(8,6))

sns.violinplot(data= raw_data, x = 'gender', y = 'salary')

plt.xlabel('Gender', fontsize = 14)

plt.ylabel('Salary', fontsize = 14)

plt.title('Salary distribution based on Gender', fontsize = 18)

plt.show()
fig = plt.figure(figsize=(8,6))

sns.countplot(data= raw_data, hue = 'degree_t', x = 'status')

plt.xlabel('Status', fontsize = 14)

plt.ylabel('Count', fontsize = 14)

plt.title('Placement Rate by Field of Degree', fontsize = 18)

plt.show()
fig = plt.figure(figsize=(8,6))

sns.violinplot(data= raw_data, x = 'degree_t', y = 'salary')

plt.xlabel('Gender', fontsize = 14)

plt.ylabel('Salary', fontsize = 14)

plt.title('Salary distribution by on Field of Degree', fontsize = 18)

plt.show()
fig = plt.figure(figsize=(8,6))

sns.countplot(data= raw_data, hue = 'specialisation', x = 'status')

plt.xlabel('Status', fontsize = 14)

plt.ylabel('Count', fontsize = 14)

plt.title('Placement Rate by MBA Specialisation', fontsize = 18)

plt.show()
fig = plt.figure(figsize=(8,6))

sns.violinplot(data= raw_data, x = 'specialisation', y = 'salary')

plt.xlabel('Gender', fontsize = 14)

plt.ylabel('Salary', fontsize = 14)

plt.title('Salary distribution based on MBA specialisation', fontsize = 18)

plt.show()
placement_data = raw_data.copy()

placement_data.head()
X = placement_data.drop(['sl_no','status','salary'], axis = 1)

y = placement_data['status']
#Machine Learning models take just numbers so any string values we have in our data will have to be converted to numbers.



#Using Column Transformer and One Hot Encoder rather than Label Encoder and One Hot Encoder as both give the same results.

#Using this method is however more effcient since i use just two lines of code.



#One Hot Encoder sorts the values for each column in ascending order and encodes each category based on this order. Eg male and 

#female, female will have a value of 1, 0 and male 0, 1. The output from One Hot Encoding puts the encoded columns first and 

#then the other columns that were not encoded.



ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 2, 4, 5, 7, 8, 10])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
print(X[:1])
lab_enc = LabelEncoder()

y = lab_enc.fit_transform(y)
print(y[:5])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
models = [LogisticRegression(max_iter = 1500), 

          KNeighborsClassifier(),

          SVC(kernel = 'linear'), 

          SVC(kernel = 'rbf'), 

          GaussianNB(), 

          DecisionTreeClassifier(), 

          RandomForestClassifier(), 

          XGBClassifier(),

          LGBMClassifier(),

          ExtraTreesClassifier()]



a, b, c, d = [], [], [], []



for i in models:

    model = i.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    

    a.append(accuracy_score(y_test, y_pred))

    b.append(f1_score(y_test, y_pred))

    c.append(precision_score(y_test, y_pred))

    d.append(recall_score(y_test, y_pred))

    

class_metrics = pd.DataFrame([a, b, c, d], index = ['Accuracy','F1 Score','Precision','Recall'], 

                             columns = ['Logistic Reg','KNN','SVM','KSVM','Naive Bayes','Decision Tree','Random Forest', 

                                        'XGBoost','LGBM','Extra Trees'])



class_metrics.transpose().sort_values(by='Accuracy', ascending=False)
log_classifier = LogisticRegression(max_iter = 1500)

log_classifier.fit(X_train, y_train)
log_pred = log_classifier.predict(X_test)
log_cm = confusion_matrix(y_test, log_pred)

print(log_cm)

accuracy_score(y_test, log_pred)
print(classification_report(y_test, log_pred))
accuracies = cross_val_score(estimator = log_classifier, X = X_train, y = y_train, cv = 10)



print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
log_classifier.get_params()
importance = log_classifier.coef_[0]

for i, v in enumerate(importance):

    print('Feature: %0d, Score:%.5f' % (i, v))

#plotting feature importance

plt.bar([x for x in range(len(importance))], importance)

plt.show()
placement_data.head(1)
print(X[:1])