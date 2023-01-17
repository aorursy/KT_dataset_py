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
# Importing additional libraries

import math



# Visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")



#Model Metric libraries

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



import warnings

warnings.filterwarnings("ignore")
# Reading the input files

raw_train = pd.read_csv('../input/titanic/train.csv')

raw_test = pd.read_csv('../input/titanic/test.csv')
raw_train.head(3)
raw_test.head(3)
df_train = raw_train[['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked','Survived']]

df_test = raw_test[['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]
df_train.info()

print('*' * 50)

df_test.info()
df_combined = pd.concat([df_train, df_test])

df_combined.info()
df_combined.groupby('Embarked')['Embarked'].count()
df_train['Age'].fillna(df_combined['Age'].median(), inplace= True)

df_test['Age'].fillna(df_combined['Age'].median(), inplace= True)

df_test['Fare'].fillna(df_combined['Fare'].median(), inplace= True)

df_train['Embarked'].fillna('S', inplace= True)

df_combined['Age'].fillna(df_combined['Age'].median(), inplace= True)

df_combined['Fare'].fillna(df_combined['Fare'].median(), inplace= True)

df_combined['Embarked'].fillna('S', inplace= True)
df_train.info()

print('*' * 50)

df_test.info()
df_train['Cabin'].unique()
df_train['Cabin'].fillna('No Cabin', inplace= True)

df_test['Cabin'].fillna('No Cabin', inplace= True)

df_combined['Cabin'].fillna('No Cabin', inplace= True)



def isCabin(x):

     return 0 if x == 'No Cabin' else 1
df_train['isCabin'] = df_train['Cabin'].apply(isCabin)

df_train.head(3)
df_test['isCabin'] = df_test['Cabin'].apply(isCabin)

df_test.tail(3)
df_combined['isCabin'] = df_combined['Cabin'].apply(isCabin)

df_combined.tail(3)
df_train.info()

print('*' * 50)

df_test.info()
for data in df_train:

    df_train['Prefix'] = df_train['Name'].str.extract(' ([A-Za-z]+)\.')



for data in df_train:

    df_test['Prefix'] = df_test['Name'].str.extract(' ([A-Za-z]+)\.')

    

for data in df_combined:

    df_combined['Prefix'] = df_combined['Name'].str.extract(' ([A-Za-z]+)\.')
df_train.head(3)
df_test.head(3)
df_train['isAlone'] = np.where((df_train['SibSp'] == 0) & 

                               (df_train['Parch'] == 0),1, 0)

df_test['isAlone'] = np.where((df_test['SibSp'] == 0) & 

                               (df_test['Parch'] == 0),1, 0)

df_combined['isAlone'] = np.where((df_combined['SibSp'] == 0) & 

                               (df_combined['Parch'] == 0),1, 0)
df_train.head(3)
fig = plt.figure(figsize=(20,20))

cols = 4

rows = math.ceil(float(df_combined.shape[1])/cols)



for i, column in enumerate(['Pclass','Prefix','Sex','Age','isAlone','Fare','isCabin','Embarked']):

  ax = fig.add_subplot(rows, cols, i+1)

  ax.set_title(column)

  if df_combined.dtypes[column] == np.object:

    df_combined[column].value_counts().plot(kind='bar', axes=ax)

  else:

    df_combined[column].hist(axes=ax)

    plt.xticks(rotation='vertical')

plt.subplots_adjust(hspace=0.7,wspace=0.2)

plt.show()
fig = plt.gcf();

fig.set_size_inches(15, 6)

sns.countplot(x='Survived', data=df_combined, hue='Prefix');
print("Male Survival Rate:", round(df_combined['Survived'][df_combined['Sex'] == 'male'].value_counts(normalize = True)[1]*100,2))

print("Females Survival Rate:", round(df_combined['Survived'][df_combined['Sex'] == 'female'].value_counts(normalize = True)[1]*100,2))
fig = plt.gcf();

fig.set_size_inches(12, 7)

sns.countplot(x='Survived', data=df_combined, hue='Sex');
print('S Survival Rate:', round(df_combined['Survived'][df_combined['Embarked'] == 'S'].value_counts(normalize = True)[1]*100,2))

print('C Survival Rate:', round(df_combined['Survived'][df_combined['Embarked'] == 'C'].value_counts(normalize = True)[1]*100,2))

print('Q Survival Rate:', round(df_combined['Survived'][df_combined['Embarked'] == 'Q'].value_counts(normalize = True)[1]*100,2))
fig = plt.gcf();

fig.set_size_inches(12, 7)

sns.countplot(x='Survived', data=df_combined, hue='Embarked')
useful_features = df_combined[['Pclass','Age','isAlone','Fare','isCabin','Survived']]



fig = plt.gcf();

fig.set_size_inches(12, 7)

sns.heatmap(useful_features.corr(), annot=True, cmap="YlGnBu");
g = df_combined[['Sex','Age']].groupby('Sex')
fig = plt.gcf();

fig.set_size_inches(12, 7)

bins = [0, 1, 10, 18, 35, 60, 80]

sns.distplot(g.get_group('female')['Age'], color="skyblue", bins= bins)

sns.distplot(g.get_group('male')['Age'], color="olive", bins= bins)

plt.legend(df_combined['Sex'])
def p_type(passenger):

    age, sex = passenger

    if age < 18:

        return 'minor'

    else:

        return sex
df_train['Person'] = df_train[['Age','Sex']].apply(p_type, axis= 1)

df_test['Person'] = df_test[['Age','Sex']].apply(p_type, axis= 1)

df_combined['Person'] = df_combined[['Age','Sex']].apply(p_type, axis= 1)

df_combined['Person'].head(10)
print('Minor Survival Rate:', round(df_combined['Survived'][df_combined['Person'] == 'minor'].value_counts(normalize = True)[1]*100,2))

print('Female Survival Rate:', round(df_combined['Survived'][df_combined['Person'] == 'female'].value_counts(normalize = True)[1]*100,2))

print('Male Survival Rate:', round(df_combined['Survived'][df_combined['Person'] == 'male'].value_counts(normalize = True)[1]*100,2))
fig = plt.gcf();

fig.set_size_inches(12, 7)

sns.countplot(x='Survived', data=df_combined, hue='Person');
print("Class 1 Survival Rate:", round(df_combined['Survived'][df_combined['Pclass'] == 1].value_counts(normalize = True)[1]*100,2))

print("Class 2 Survival Rate:", round(df_combined['Survived'][df_combined['Pclass'] == 2].value_counts(normalize = True)[1]*100,2))

print("Class 3 Survival Rate:", round(df_combined['Survived'][df_combined['Pclass'] == 3].value_counts(normalize = True)[1]*100,2))
fig = plt.gcf();

fig.set_size_inches(12, 7)

sns.countplot(x='Survived', data=df_combined, hue='Pclass');
print("Alone Survival Rate:", round(df_combined['Survived'][df_combined['isAlone'] == 1].value_counts(normalize = True)[1]*100,2))

print("With Family Survival Rate:", round(df_combined['Survived'][df_combined['isAlone'] == 0].value_counts(normalize = True)[1]*100,2))
fig = plt.gcf();

fig.set_size_inches(12, 7)

sns.countplot(x='Survived', data=df_combined, hue='isAlone');
df_train['FareGroup'] = pd.qcut(df_train['Fare'], 4, labels = [1, 2, 3, 4])

df_test['FareGroup'] = pd.qcut(df_test['Fare'], 4, labels = [1, 2, 3, 4])

df_combined['FareGroup'] = pd.qcut(df_combined['Fare'], 4, labels = [1, 2, 3, 4])
print('FareQ1:', round(df_combined['Survived'][df_combined['FareGroup'] == 1].value_counts(normalize = True)[1]*100,2))

print('FareQ2:', round(df_combined['Survived'][df_combined['FareGroup'] == 2].value_counts(normalize = True)[1]*100,2))

print('FareQ3:', round(df_combined['Survived'][df_combined['FareGroup'] == 3].value_counts(normalize = True)[1]*100,2))

print('FareQ4:', round(df_combined['Survived'][df_combined['FareGroup'] == 4].value_counts(normalize = True)[1]*100,2))
fig = plt.gcf();

fig.set_size_inches(12, 7)

sns.countplot(x='Survived', data=df_combined, hue='FareGroup');
print('Without Cabin:', round(df_combined['Survived'][df_combined['isCabin'] == 0].value_counts(normalize = True)[1]*100,2))

print('With Cabin:', round(df_combined['Survived'][df_combined['isCabin'] == 1].value_counts(normalize = True)[1]*100,2))
fig = plt.gcf();

fig.set_size_inches(12, 7)

sns.countplot(x='Survived', data=df_combined, hue='isCabin');
fig = plt.gcf();

fig.set_size_inches(8, 5)

sns.countplot(x='Survived', data=df_combined);
df_train['FareGroup'] = pd.to_numeric(df_train.FareGroup)

df_test['FareGroup'] = pd.to_numeric(df_test.FareGroup)
df_train['FareGroup'] = pd.to_numeric(df_train.FareGroup)

df_test['FareGroup'] = pd.to_numeric(df_test.FareGroup)
df_train.drop(['Name','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked'], axis= 1, inplace= True)

df_test.drop(['Name','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked'], axis= 1, inplace= True)
df_train.info()
df_train = pd.get_dummies(df_train, columns=['Pclass','Prefix','Person','isAlone',

                                             'isCabin','FareGroup'])

df_test = pd.get_dummies(df_test, columns=['Pclass','Prefix','Person','isAlone',

                                           'isCabin','FareGroup'])
df_train.columns
df_train = df_train[['PassengerId', 'Pclass_1', 'Pclass_2',

                     'Pclass_3', 'Prefix_Capt', 'Prefix_Col', 'Prefix_Countess',

                     'Prefix_Don', 'Prefix_Dr', 'Prefix_Jonkheer', 

                     'Prefix_Lady', 'Prefix_Major', 'Prefix_Master', 

                     'Prefix_Miss', 'Prefix_Mlle', 'Prefix_Mme', 'Prefix_Mr',

                     'Prefix_Mrs', 'Prefix_Ms', 'Prefix_Rev', 'Prefix_Sir', 

                     'Person_female', 'Person_male', 'Person_minor', 'isAlone_0', 'isAlone_1', 

                     'isCabin_0', 'isCabin_1', 'FareGroup_1', 'FareGroup_2',

                     'FareGroup_3', 'FareGroup_4', 'Survived']]



df_test['Prefix_Capt'] = 0

df_test['Prefix_Countess'] = 0

df_test['Prefix_Don'] = 0

df_test['Prefix_Jonkheer'] = 0

df_test['Prefix_Lady'] = 0

df_test['Prefix_Major'] = 0

df_test['Prefix_Mlle'] = 0

df_test['Prefix_Mme'] = 0

df_test['Prefix_Ms'] = 0

df_test['Prefix_Sir'] = 0



df_test = df_test[['PassengerId', 'Pclass_1', 'Pclass_2',

                     'Pclass_3', 'Prefix_Capt', 'Prefix_Col', 'Prefix_Countess',

                     'Prefix_Don', 'Prefix_Dr', 'Prefix_Jonkheer', 

                     'Prefix_Lady', 'Prefix_Major', 'Prefix_Master', 

                     'Prefix_Miss', 'Prefix_Mlle', 'Prefix_Mme', 'Prefix_Mr',

                     'Prefix_Mrs', 'Prefix_Ms', 'Prefix_Rev', 'Prefix_Sir', 

                     'Person_female', 'Person_male', 'Person_minor','isAlone_0', 'isAlone_1', 

                     'isCabin_0', 'isCabin_1', 'FareGroup_1', 'FareGroup_2',

                     'FareGroup_3', 'FareGroup_4']]
df_train.info()

print('^'*50)

df_test.info()
df_train.shape
df_test.shape
# split features (X) & target (y)

X = df_train.drop(['PassengerId','Survived'], axis=1)

y = df_train['Survived']



print(X.shape)

print(y.shape)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 5)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()

gnb.fit(X_train,y_train)
y_pred_trn_gnb = gnb.predict(X_train)

y_pred_val_gnb = gnb.predict(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



print('Gaussian NB Performance')

print('--' * 50)

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_val_gnb))

print('--' * 50)

target_names = ['Deceased', 'Survived']

print('Classification Report')

print(classification_report(y_test, y_pred_val_gnb, target_names= target_names))

print('--' * 50)

acc_gnb_trn = round(accuracy_score(y_train, y_pred_trn_gnb)*100,2)

acc_gnb_val = round(accuracy_score(y_test, y_pred_val_gnb)*100,2)

print('Training Accuracy: ', acc_gnb_trn)

print('Validation Accuracy: ', acc_gnb_val)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors= 9, weights= 'distance')

knn.fit(X_train,y_train)
y_pred_trn_knn = knn.predict(X_train)

y_pred_val_knn = knn.predict(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



print('KNN Performance')

print('--' * 50)

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_val_knn))

print('--' * 50)

target_names = ['Deceased', 'Survived']

print('Classification Report')

print(classification_report(y_test, y_pred_val_knn, target_names= target_names))

print('--' * 50)

acc_knn_trn = round(accuracy_score(y_train, y_pred_trn_knn)*100,2)

acc_knn_val = round(accuracy_score(y_test, y_pred_val_knn)*100,2)

print('Training Accuracy: ', acc_knn_trn)

print('Validation Accuracy: ', acc_knn_val)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train,y_train)
y_pred_trn_logreg = logreg.predict(X_train)

y_pred_val_logreg = logreg.predict(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



print('Logistic Regression Performance')

print('--' * 50)

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_val_logreg))

print('--' * 50)

target_names = ['Deceased', 'Survived']

print('Classification Report')

print(classification_report(y_test, y_pred_val_logreg, target_names= target_names))

print('--' * 50)

acc_logreg_trn = round(accuracy_score(y_train, y_pred_trn_logreg)*100,2)

acc_logreg_val = round(accuracy_score(y_test, y_pred_val_logreg)*100,2)

print('Training Accuracy: ', acc_logreg_trn)

print('Validation Accuracy: ', acc_logreg_val)
from sklearn.svm import SVC



svc = SVC(kernel= 'poly', probability= True)

svc.fit(X_train,y_train)
y_pred_trn_svc = svc.predict(X_train)

y_pred_val_svc = svc.predict(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



print('SVC Performance')

print('--' * 50)

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_val_svc))

print('--' * 50)

target_names = ['Deceased', 'Survived']

print('Classification Report')

print(classification_report(y_test, y_pred_val_svc, target_names= target_names))

print('--' * 50)

acc_svc_trn = round(accuracy_score(y_train, y_pred_trn_svc)*100,2)

acc_svc_val = round(accuracy_score(y_test, y_pred_val_svc)*100,2)

print('Training Accuracy: ', acc_svc_trn)

print('Validation Accuracy: ', acc_svc_val)
from sklearn.tree import DecisionTreeClassifier



dtree = DecisionTreeClassifier(criterion= 'gini', 

                               max_depth= 7)

dtree.fit(X_train,y_train)
y_pred_trn_dtree = dtree.predict(X_train)

y_pred_val_dtree = dtree.predict(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



print('Decision Tree Performance')

print('--' * 50)

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_val_dtree))

print('--' * 50)

target_names = ['Deceased', 'Survived']

print('Classification Report')

print(classification_report(y_test, y_pred_val_dtree, target_names= target_names))

print('--' * 50)

acc_dtree_trn = round(accuracy_score(y_train, y_pred_trn_dtree)*100,2)

acc_dtree_val = round(accuracy_score(y_test, y_pred_val_dtree)*100,2)

print('Training Accuracy: ', acc_dtree_trn)

print('Validation Accuracy: ', acc_dtree_val)
from sklearn.ensemble import RandomForestClassifier



rndf = RandomForestClassifier(n_estimators= 50,

                              criterion= 'entropy',

                              max_depth= 7,

                              max_features= 5,

                              random_state= 5)

rndf.fit(X_train,y_train)
y_pred_trn_rndf = rndf.predict(X_train)

y_pred_val_rndf = rndf.predict(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



print('Random Forest Performance')

print('--' * 50)

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_val_rndf))

print('--' * 50)

target_names = ['Deceased', 'Survived']

print('Classification Report')

print(classification_report(y_test, y_pred_val_rndf, target_names= target_names))

print('--' * 50)

acc_rndf_trn = round(accuracy_score(y_train, y_pred_trn_rndf)*100,2)

acc_rndf_val = round(accuracy_score(y_test, y_pred_val_rndf)*100,2)

print('Training Accuracy: ', acc_rndf_trn)

print('Validation Accuracy: ', acc_rndf_val)
from sklearn.ensemble import AdaBoostClassifier



adab = AdaBoostClassifier()

adab.fit(X_train,y_train)
y_pred_trn_adab = adab.predict(X_train)

y_pred_val_adab = adab.predict(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



print('AdaBoost Performance')

print('--' * 50)

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_val_adab))

print('--' * 50)

target_names = ['Deceased', 'Survived']

print('Classification Report')

print(classification_report(y_test, y_pred_val_adab, target_names= target_names))

print('--' * 50)

acc_adab_trn = round(accuracy_score(y_train, y_pred_trn_adab)*100,2)

acc_adab_val = round(accuracy_score(y_test, y_pred_val_adab)*100,2)

print('Training Accuracy: ', acc_adab_trn)

print('Validation Accuracy: ', acc_adab_val)
from sklearn.ensemble import GradientBoostingClassifier



gbc = GradientBoostingClassifier(learning_rate= 0.2,

                                 n_estimators= 300,

                                 max_depth= 4)

gbc.fit(X_train,y_train)
y_pred_trn_gbc = gbc.predict(X_train)

y_pred_val_gbc = gbc.predict(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



print('Gradient Boosting Classifier Performance')

print('--' * 50)

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_val_gbc))

print('--' * 50)

target_names = ['Deceased', 'Survived']

print('Classification Report')

print(classification_report(y_test, y_pred_val_gbc, target_names= target_names))

print('--' * 50)

acc_gbc_trn = round(accuracy_score(y_train, y_pred_trn_gbc)*100,2)

acc_gbc_val = round(accuracy_score(y_test, y_pred_val_gbc)*100,2)

print('Training Accuracy: ', acc_gbc_trn)

print('Validation Accuracy: ', acc_gbc_val)
import xgboost



xgb = xgboost.XGBClassifier(learning_rate= 0.16,

                            subsample= 0.5,

                            max_depth= 4)

xgb.fit(X_train,y_train)
y_pred_trn_xgb = xgb.predict(X_train)

y_pred_val_xgb = xgb.predict(X_test)
print('XGB Performance')

print('--' * 50)

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_val_xgb))

print('--' * 50)

target_names = ['Deceased', 'Survived']

print('Classification Report')

print(classification_report(y_test, y_pred_val_xgb, target_names= target_names))

print('--' * 50)

acc_xgb_trn = round(accuracy_score(y_train, y_pred_trn_xgb)*100,2)

acc_xgb_val = round(accuracy_score(y_test, y_pred_val_xgb)*100,2)

print('Training Accuracy: ', acc_xgb_trn)

print('Validation Accuracy: ', acc_xgb_val)
best_model = pd.DataFrame({

    'Model': ['Gaussian NB', 'KNN', 'Logistic Reg.', 'SVM', 'Decision Tree',

              'Random Forest', 'AdaBoost', 'Gradient Boost', 'XGBoost'],

    'Training Accuracy': [acc_gnb_trn, acc_knn_trn, acc_logreg_trn, acc_svc_trn, acc_dtree_trn,

                 acc_rndf_trn, acc_adab_trn, acc_gbc_trn, acc_xgb_trn],

    'Validation Accuracy': [acc_gnb_val, acc_knn_val, acc_logreg_val, acc_svc_val, acc_dtree_val,

                 acc_rndf_val, acc_adab_val, acc_gbc_val, acc_xgb_val]})



best_model.sort_values(by= 'Validation Accuracy', ascending= False)
# Importing libraries for plotting roc_curve

from sklearn.metrics import roc_curve, roc_auc_score
classifiers = [GaussianNB(),

               KNeighborsClassifier(n_neighbors= 9, weights= 'distance'),

               LogisticRegression(),               

               SVC(kernel= 'poly', probability= True),                 

               DecisionTreeClassifier(criterion= 'gini', 

                                      max_depth= 7),

               RandomForestClassifier(n_estimators= 50,

                                      criterion= 'entropy',

                                      max_depth= 7,

                                      max_features= 5,

                                      random_state= 5),

               AdaBoostClassifier(),

               GradientBoostingClassifier(learning_rate= 0.2,

                                          n_estimators= 300,

                                          max_depth= 4),

               xgboost.XGBClassifier(learning_rate= 0.16,

                                     subsample= 0.5,

                                     max_depth= 4)]

result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

for cls in classifiers:

    model = cls.fit(X_train, y_train)

    yproba = model.predict_proba(X_test)[:,1]

    

    fpr, tpr, _ = roc_curve(y_test,  yproba)

    auc = roc_auc_score(y_test, yproba)

    

    result_table = result_table.append({'classifiers':cls.__class__.__name__,

                                        'fpr':fpr, 

                                        'tpr':tpr, 

                                        'auc':auc}, ignore_index=True)

result_table.set_index('classifiers', inplace=True)
fig = plt.gcf()

fig.set_size_inches(9, 6)



for i in result_table.index:

    plt.plot(result_table.loc[i]['fpr'], 

             result_table.loc[i]['tpr'], 

             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    

plt.plot([0,1], [0,1], color= 'red', linestyle='-')



plt.xticks(np.arange(0.0, 1.1, step=0.1))

plt.xlabel('False Positive Rate')



plt.yticks(np.arange(0.0, 1.1, step=0.1))

plt.ylabel('True Positive Rate')



plt.title('AUC - ROC Curve Analysis', fontweight= 'bold', fontsize= 15)

plt.legend(prop={'size':11}, loc='lower right')



plt.show()
# Set ids as PassengerId and predict survival 

ids = df_test['PassengerId']

test_file = df_test.drop(['PassengerId'], axis= 1)

final_pred = xgb.predict(test_file)



# Set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({'PassengerId' : ids, 'Survived': final_pred})

output.to_csv('submission.csv', index=False)

print(output)