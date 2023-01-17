import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

import warnings
warnings.filterwarnings("ignore")
covid_data = pd.read_excel('../input/covid19/dataset.xlsx')
covid_data.head()
covid_data.shape
covid_data[covid_data['SARS-Cov-2 exam result']=='positive'].shape[0]
positive_data = covid_data[covid_data['SARS-Cov-2 exam result']=='positive']
sns.countplot(x='Patient age quantile', data=positive_data)
positive_data['Patient age quantile'].mean()
sns.countplot(x='SARS-Cov-2 exam result', data=covid_data, )
sns.countplot(x='Patient addmited to regular ward (1=yes, 0=no)', data=positive_data)
sns.countplot(x='Patient addmited to semi-intensive unit (1=yes, 0=no)', data=positive_data)
sns.countplot(x='Patient addmited to intensive care unit (1=yes, 0=no)', data=positive_data)
covid_data = covid_data.fillna(0)
results = pd.get_dummies(covid_data['SARS-Cov-2 exam result'])
type_urine = pd.get_dummies(covid_data['Urine - Aspect'])
cristais_urine = pd.get_dummies(covid_data['Urine - Crystals'])
color_urine = pd.get_dummies(covid_data['Urine - Color'])
cristais_urine = cristais_urine.drop([0], axis=1)
type_urine = color_urine.drop([0], axis=1)
color_urine = color_urine.drop([0], axis=1)
covid_data = pd.concat([covid_data,type_urine, cristais_urine, color_urine], axis=1)
covid_data = covid_data.drop(['Patient ID', 'Urine - Aspect', 'Urine - Crystals', 'Urine - Color'], axis=1)
covid_data = covid_data.replace("not_detected", 0)
covid_data = covid_data.replace("detected", 1)
covid_data = covid_data.replace("negative", 0)
covid_data = covid_data.replace("positive", 1)
covid_data = covid_data.replace("not_done", 0)
covid_data = covid_data.replace("done", 1)
covid_data = covid_data.replace("absent", 0)
covid_data = covid_data.replace("Ausentes", 0)
covid_data = covid_data.replace("present", 1)
covid_data = covid_data.replace("normal", 1)
covid_data = covid_data.replace("<1000", 999)
covid_data = covid_data.replace("Não Realizado", 0)
covid_data['SARS-Cov-2 exam result'] = results['positive']
X = covid_data.drop(['SARS-Cov-2 exam result'], axis=1)
y = covid_data['SARS-Cov-2 exam result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
rfc = RandomForestClassifier(n_estimators=600, max_depth=2, class_weight="balanced")
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
svc_model = SVC(class_weight='balanced')
svc_model.fit(X_train,y_train)
predictions = svc_model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
covid_data['Triagem'] = covid_data.apply(lambda row: 1 if row['Patient addmited to regular ward (1=yes, 0=no)']==1 and row['Patient addmited to semi-intensive unit (1=yes, 0=no)']==0 and row['Patient addmited to intensive care unit (1=yes, 0=no)']==0 else 0, axis=1)
covid_data['Triagem'] = covid_data.apply(lambda row: 2 if row['Triagem']==0 and row['Patient addmited to regular ward (1=yes, 0=no)']==0 and row['Patient addmited to semi-intensive unit (1=yes, 0=no)']==1 and row['Patient addmited to intensive care unit (1=yes, 0=no)']==0 else row['Triagem'], axis=1)
covid_data['Triagem'] = covid_data.apply(lambda row: 3 if row['Triagem']==0 and row['Patient addmited to regular ward (1=yes, 0=no)']==0 and row['Patient addmited to semi-intensive unit (1=yes, 0=no)']==0 and row['Patient addmited to intensive care unit (1=yes, 0=no)']==1 else row['Triagem'], axis=1)
covid_data = covid_data.drop(['Patient addmited to regular ward (1=yes, 0=no)','Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)'], axis=1)
X = covid_data.drop(['Triagem'], axis=1)
y = covid_data['Triagem']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
rfc = RandomForestClassifier(n_estimators=600, max_depth=2, class_weight="balanced")
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
svc_model = SVC(class_weight='balanced')
svc_model.fit(X_train,y_train)
predictions = svc_model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))