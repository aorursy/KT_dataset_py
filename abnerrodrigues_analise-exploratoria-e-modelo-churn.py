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
import pandas_profiling as pp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, roc_curve, auc, confusion_matrix, plot_roc_curve, accuracy_score
from sklearn import preprocessing

raw_data = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(raw_data.dtypes)
pd.set_option('display.max_columns', None)
raw_data.head()
raw_data.isnull().sum()
cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod']

plt.subplots(4, 4, sharex=True, sharey=True, figsize=(15,15))
i = 1
for column in cols:
    plt.subplot(4, 4, i)
    plt.title(column)
    raw_data[column].value_counts(normalize=True).sort_values().plot(kind = 'barh', fontsize=9, sharex=False)
    i = i + 1
plt.show()


cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod']

plt.subplots(4, 4, sharex=True, sharey=True, figsize=(15,15))
i = 1
for column in cols:
    plt.subplot(4, 4, i)
    plt.title(column)
    raw_data[ raw_data['Churn'] == 'Yes'][column].value_counts(normalize=True).sort_values().plot(kind = 'barh', fontsize=9)
    i = i + 1
plt.show()
cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod']

fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(15,15))
#fig.subplots_adjust(left = 0.4)
i = 1
for column in cols:
    plt.subplot(4, 4, i)
    plt.title(column)
    raw_data[ raw_data['Churn'] == 'No'][column].value_counts(normalize=True).sort_values().plot(kind = 'barh', fontsize=9, sharex=True)

    i = i + 1
plt.show()
a = sns.jointplot(x="tenure", y="MonthlyCharges", data=raw_data[raw_data['Churn']=='Yes'], kind="kde", height=5)
plt.title('Churn Yes')
b = sns.jointplot(x="tenure", y="MonthlyCharges", data=raw_data[raw_data['Churn']=='No'], kind="kde", height=5)
plt.title('Churn No')
cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod']

fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(15,15))
#fig.subplots_adjust(left = 0.4)
i = 1
for column in cols:
    plt.subplot(4, 4, i)
    plt.title(column)
    raw_data[ (raw_data['Churn'] == 'No') & (raw_data['tenure'] > 40) & (raw_data['MonthlyCharges'] > 60)  ][column].value_counts(normalize=True).sort_values().plot(kind = 'barh', fontsize=9, sharex=True)

    i = i + 1
plt.show()
X_embedded = TSNE(n_components=2).fit_transform(pd.get_dummies(raw_data.loc[:, raw_data.columns != 'Churn']))
tsne_df = pd.DataFrame(data=X_embedded, columns=["X", "Y"])
tsne_df['Churn'] = raw_data['Churn']
sns.scatterplot(data=tsne_df, x="X", y="Y", hue="Churn", Alpha = 0.5)
# tem duas variáveis numericas que podemos criar categorias: tenure e MonthlyCharges, vou fazer isso mas manter também a variável numérica..
new_df = pd.DataFrame()
#coloquei 12 pra ter a informação do churn de 6 em 6 meses, talvez valha a pena aumentar (por exemplo pra 3 em 3 meses)
new_df['tenure_cat'] = pd.cut(raw_data['tenure'], 12, labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'] ) 
#pelo gráfico da densidade do ChurnxTenurexCusto parece ter 3 ou 4 categorias do MonthlyCharges, talvez valesse a pena fazer um ajuste fino maior, mas vou cortar em 5
new_df['charges_cat'] = pd.cut(raw_data['MonthlyCharges'], 5, labels=['1', '2', '3', '4', '5'] ) 

# essas colunas possuem um valor repetido em todas, que é o valor No Internet Service, mas esse valor já está descrito na coluna InternetService
# então eu separei as colunas que já possuem essa informação de No Internet Service repetido para remove-las depois do dummies
col_dummies_repeated = ['OnlineSecurity', 'OnlineBackup',  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies' ]
dummies = pd.get_dummies( raw_data[col_dummies_repeated ] ) 
dummies = dummies.drop( columns=['OnlineSecurity_No internet service', 'OnlineBackup_No internet service', 'DeviceProtection_No internet service', 'TechSupport_No internet service', 'StreamingTV_No internet service', 'StreamingMovies_No internet service'] )

col_dummies = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract', 'PaperlessBilling', 'PaymentMethod']
dummies2 = pd.get_dummies( raw_data[col_dummies ], drop_first=True)

#concatenando os dummies
new_df = pd.concat([new_df, dummies], axis=1)
new_df = pd.concat([new_df, dummies2], axis=1)

#pegando os dummies das variaveis categoricas que foram criadas
new_df = pd.get_dummies(new_df, drop_first=True )

#adicionando as duas colunas numericas

new_df['tenure'] = raw_data['tenure']
new_df['MonthlyCharges'] = raw_data['MonthlyCharges']

new_df['Churn'] = raw_data['Churn']
new_df = new_df.replace({'Yes': 1, 'No': 0})

new_df.head()
X_train, X_test, y_train, y_test = train_test_split( new_df.loc[:, new_df.columns != 'Churn'], new_df['Churn'], test_size=0.2, random_state=42)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

parameters = {'n_estimators':[10, 20, 50, 100], 'criterion':['gini', 'entropy'], 'max_features':['auto', 'sqrt', 'log2'] }
rf = RandomForestClassifier()
clf = GridSearchCV(rf, parameters, n_jobs=4)
clf.fit(X_train, y_train)
print('Best Params:', clf.best_params_)
y_pred = clf.predict(X_test)
m = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(clf, X_test, y_test)
clf_score = cross_val_score(rf, X_train, y_train, cv=10)
print(clf_score)
clf_score.mean()
features = pd.DataFrame(clf.best_estimator_.feature_importances_)
features["Feature"] = list(new_df.columns[:-1]) 
features.sort_values(by=0, ascending=False).head()


g = sns.barplot(0,"Feature",data = features.sort_values(by=0, ascending=False)[0:10], palette="Pastel1",orient = "h")
g.set_xlabel("Weight")
g = g.set_title("Random Forest")
parameters = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'C':[0.9, 1.0, 1.1] }
svm = SVC()
clf = GridSearchCV(svm, parameters, n_jobs=4)
clf.fit(X_train, y_train)
print('Best Params:', clf.best_params_)
y_pred = clf.predict(X_test)
m = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(clf, X_test, y_test)
clf_score = cross_val_score(svm, X_train, y_train, cv=10)
print(clf_score)
clf_score.mean()
parameters = {'penalty':['l1', 'l2', 'elasticnet', 'none'], 'C':[0.9, 1.0, 1.1] }
reglog = LogisticRegression()
clf = GridSearchCV(reglog, parameters, n_jobs=4)
clf.fit(X_train, y_train)
print('Best Params:', clf.best_params_)
y_pred = clf.predict(X_test)
m = confusion_matrix(y_test, y_pred)
clf_score = cross_val_score(reglog, X_train, y_train, cv=10)
print(clf_score)
clf_score.mean()
plot_confusion_matrix(clf, X_test, y_test)
pred_proba_df = pd.DataFrame(clf.best_estimator_.predict_proba(X_test))
threshold_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,.7,.75,.8,.85,.9,.95,.99]
for i in threshold_list:
    print ('\n******** For i = {} ******'.format(i))
    Y_test_pred = pred_proba_df.applymap(lambda x: 1 if x>i else 0)
    test_accuracy = accuracy_score(y_test, Y_test_pred.iloc[:,1].to_numpy())
    print('Our testing accuracy is {}'.format(test_accuracy))

    print(confusion_matrix(y_test, Y_test_pred.iloc[:,1].to_numpy()))
