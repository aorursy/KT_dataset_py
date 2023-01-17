import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)



import os



DATA_DIR='../input'

print(os.listdir(DATA_DIR))
# Dummy variables to hold dataset file names on my local machine

TRAIN_CSV_FILE = "../input/train.csv"

TEST_CSV_FILE = "../input/test.csv"

df_train = pd.read_csv(TRAIN_CSV_FILE)

df_test = pd.read_csv(TEST_CSV_FILE)
df_train.head()
df_train.info()
df_test.head()
df_test.info()
df_train.isna().sum()
df_test.isna().sum()
df_train.describe()
df_test.describe()
sns.countplot(data=df_train, x='Survived')
df_train.groupby(['Survived', 'Sex'])['Survived'].count()
sns.catplot(x='Sex', col='Survived', kind='count', data=df_train)
women_survived = df_train[df_train.Sex == 'female'].Survived.sum()

men_survived = df_train[df_train.Sex == 'male'].Survived.sum()

total_female_survived = df_train[df_train.Sex == 'female'].Survived.count()

total_male_survived = df_train[df_train.Sex == 'male'].Survived.count()

print(women_survived,men_survived,total_female_survived, total_male_survived, sep=' ')

print('Women Survived --> {:<7.3f}%'.format(women_survived/total_female_survived * 100))

print('Men Survived --> {:<7.3f}%'.format(men_survived/total_male_survived * 100))
f,ax=plt.subplots(1,2,figsize=(16,7))

df_train['Survived'][df_train['Sex']=='male'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[0],shadow=True)

df_train['Survived'][df_train['Sex']=='female'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[0].set_title('Survived (male)')

ax[1].set_title('Survived (female)')



plt.show()
df_train.groupby(['Survived', 'Pclass'])['Survived'].count()
sns.catplot(x='Pclass', col='Survived', kind='count', data=df_train)
pd.crosstab(df_train.Pclass, df_train.Survived, margins=True).style.background_gradient(cmap='autumn_r')
pd_class_p = pd.crosstab(df_train.Pclass, df_train.Survived, margins=True,normalize='index')

pd_class_p
pd_class_p[1][3]
print('Survivals per class percentages :')

for i in range(3):

    print('Class {} --> {:<7.3f}%'.format(i+1,pd_class_p[1][i+1]*100))
sns.catplot('Pclass','Survived', kind='point', data=df_train)
pd.crosstab([df_train.Sex, df_train.Survived], df_train.Pclass, margins=True).style.background_gradient(cmap='autumn_r')
cl_sex_sur_per = pd.crosstab([df_train.Sex, df_train.Survived], df_train.Pclass, margins=True)

cl_sex_sur_per
sns.catplot('Pclass','Survived',hue='Sex', kind='point', data=df_train);
sns.catplot(x='Survived', col='Embarked', kind='count', data=df_train);
sns.catplot('Embarked','Survived', kind='point', data=df_train);
sns.catplot(x='Sex',y='Survived', col='Embarked', kind='bar', data=df_train)
sns.catplot('Embarked','Survived', hue= 'Sex', kind='point', data=df_train);
sns.catplot('Embarked','Survived', col='Pclass', hue= 'Sex', kind='point', data=df_train)
pd.crosstab([df_train.Survived], [df_train.Sex, df_train.Pclass, df_train.Embarked], margins=True)
for df in [df_train, df_test]:

    df['Age_bin']=np.nan

    for i in range(8,0,-1):

        df.loc[ df['Age'] <= i*10, 'Age_bin'] = i
df_train[['Age', 'Age_bin']].head(20)
sns.catplot(x='Age_bin',y='Survived',  kind='bar', data=df_train)
sns.catplot(x='Age_bin',y='Survived',col='Sex',  kind='bar', data=df_train)
sns.catplot('Age_bin','Survived',hue='Sex',kind='point',data=df_train)
sns.catplot('Age_bin','Survived', col='Pclass', row = 'Sex', kind='point', data=df_train);
pd.crosstab([df_train.Sex, df_train.Survived], [df_train.Age_bin, df_train.Pclass], margins=True).style.background_gradient(cmap='autumn_r')
sns.catplot('SibSp','Survived', col='Pclass' , row = 'Sex', kind='point', data=df_train)
pd.crosstab([df_train.Sex, df_train.Survived], [df_train.SibSp, df_train.Pclass], margins=True).style.background_gradient(cmap='autumn_r')
sns.catplot('Parch','Survived', col='Pclass' , row = 'Sex', kind='point', data=df_train)
pd.crosstab([df_train.Sex, df_train.Survived], [df_train.Parch, df_train.Pclass], margins=True).style.background_gradient(cmap='autumn_r')
sns.distplot(df_train['Fare'])
for df in [df_train, df_test]:

    df['Fare_bin']=np.nan

    for i in range(12,0,-1):

        df.loc[ df['Fare'] <= i*50, 'Fare_bin'] = i
df_train['Fare_bin'].head(10)
sns.catplot(x='Fare_bin',y='Survived',col='Sex',  kind='bar', data=df_train)
sns.catplot('Fare_bin','Survived', col='Pclass' , row = 'Sex', kind='point', data=df_train)
pd.crosstab([df_train.Sex, df_train.Survived], [df_train.Fare_bin, df_train.Pclass], margins=True).style.background_gradient(cmap='autumn_r')
df_train_ml = pd.read_csv(TRAIN_CSV_FILE)

df_test_ml = pd.read_csv(TEST_CSV_FILE)
df_train_ml.head()
df_test_ml.head()
# Encoding categorical data

df_train_ml = pd.get_dummies(data=df_train_ml, columns=['Sex', 'Embarked'], drop_first=True)

df_train_ml.drop(['Name','Ticket', 'Cabin'],axis=1, inplace=True) 



passenger_id = df_test_ml['PassengerId']

df_test_ml = pd.get_dummies(data=df_test_ml, columns=['Sex', 'Embarked'], drop_first=True)

df_test_ml.drop(['Name','Ticket', 'Cabin'],axis=1, inplace=True) 
df_train_ml.head()
df_test_ml.head()
X = df_train_ml.iloc[:, 2:].values

y = df_train_ml.iloc[:, 1].values
# Taking care of missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(X[:, 1:2])

X[:, 1:2] = imputer.transform(X[:, 1:2])
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
X_train_all = X

y_train_all = y

X_test_all = df_test_ml.iloc[:,1:].values
# Take care of NaNs in all data

imputer = imputer.fit(X_test_all[:, [1,4]])

X_test_all[:, [1,4]] = imputer.transform(X_test_all[:, [1,4]])
sc_all = StandardScaler()

X_train_all = sc_all.fit_transform(X_train_all)

X_test_all = sc_all.transform(X_test_all)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report, accuracy_score



def show_metrics(y_test, y_pred,msg='Summary'):

    cm = confusion_matrix(y_test,y_pred)

    cm = sns.heatmap(cm, annot=True, fmt='d')

    print(msg)

    print(classification_report(y_test, y_pred))

    print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))
from sklearn.linear_model import LogisticRegression

lg_classifier = LogisticRegression(random_state = 101)

lg_classifier.fit(X_train, y_train)
# Predicting the Test set results

lg_y_pred = lg_classifier.predict(X_test)
# Making the Confusion Matrix

cm = confusion_matrix(y_test, lg_y_pred)

sns.heatmap(cm, annot=True, fmt='d')
# Print some metrics

print(classification_report(y_test, lg_y_pred))

print(accuracy_score(y_test, lg_y_pred))
lg_classifier.fit(X_train_all, y_train_all)

lg_y_pred_all = lg_classifier.predict(X_test_all)
sub_logreg = pd.DataFrame()

sub_logreg['PassengerId'] = df_test['PassengerId']

sub_logreg['Survived'] = lg_y_pred_all

#sub_logmodel.to_csv('logmodel.csv',index=False)
# Fitting K-NN to the Training set

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

knn_classifier.fit(X_train, y_train)
# Predicting the Test set results

knn_y_pred = knn_classifier.predict(X_test)
#confusion matrix

knn_cm = confusion_matrix(y_test, knn_y_pred)

sns.heatmap(knn_cm, annot=True, fmt='d')
print('K-NN Summary')

print(classification_report(y_test, knn_y_pred))

print(accuracy_score(y_test, knn_y_pred))
knn_classifier.fit(X_train_all, y_train_all)

knn_y_pred_all= knn_classifier.predict(X_test_all)
sub_knn = pd.DataFrame()

sub_knn['PassengerId'] = df_test['PassengerId']

sub_knn['Survived'] = knn_y_pred_all

#sub_knn.to_csv('knn.csv',index=False)
# Fitting SVM to the Training set

from sklearn.svm import SVC

svm_classifier = SVC(kernel = 'linear', random_state = 101)

svm_classifier.fit(X_train, y_train)
# Predicting the Test set results

svm_y_pred = svm_classifier.predict(X_test)
#confusion matrix

svm_cm = confusion_matrix(y_test, svm_y_pred)

sns.heatmap(svm_cm, annot=True, fmt='d')

print('SVM Summary')

print(classification_report(y_test, svm_y_pred))

print(accuracy_score(y_test, svm_y_pred))
svm_classifier.fit(X_train_all, y_train_all)

svm_y_pred_all= svm_classifier.predict(X_test_all)
sub_svm = pd.DataFrame()

sub_svm['PassengerId'] = df_test['PassengerId']

sub_svm['Survived'] = svm_y_pred_all

#sub_svm.to_csv('svm.csv',index=False)
# Fitting Kernel SVM to the Training set

ksvm_classifier = SVC(kernel = 'rbf', random_state = 101)

ksvm_classifier.fit(X_train, y_train)
# Predicting the Test set results

ksvm_y_pred = ksvm_classifier.predict(X_test)
#confusion matrix and metrics for kernel SVM

ksvm_cm = confusion_matrix(y_test, ksvm_y_pred)

sns.heatmap(ksvm_cm, annot=True, fmt='d')

print('Kernel SVM Summary')

print(classification_report(y_test, ksvm_y_pred))

print(accuracy_score(y_test, ksvm_y_pred))
ksvm_classifier.fit(X_train_all, y_train_all)

ksvm_y_pred_all= ksvm_classifier.predict(X_test_all)
sub_ksvm = pd.DataFrame()

sub_ksvm['PassengerId'] = df_test['PassengerId']

sub_ksvm['Survived'] = ksvm_y_pred_all

#sub_svm.to_csv('svm.csv',index=False)
# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

nb_classifier = GaussianNB()

nb_classifier.fit(X_train, y_train)
# Predicting the Test set results

nb_y_pred = nb_classifier.predict(X_test)
show_metrics(y_test, nb_y_pred, msg='Naives Bayes Summary')
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

rf_classifier.fit(X_train, y_train)
# Predicting the Test set results

rf_y_pred = rf_classifier.predict(X_test)
show_metrics(y_test, rf_y_pred, msg='Random Forest Summary')
# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = ksvm_classifier, X = X_train, y = y_train, cv = 10)

print(accuracies.mean())

print(accuracies.std())
# Applying Grid Search to find the best model and the best parameters

from sklearn.model_selection import GridSearchCV

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},

              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator = ksvm_classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print('Best Accuracy : {}\n'.format(best_accuracy))

print('Best Parameters : {}\n'.format(best_parameters))
# Fitting Kernel SVM to the Training set

ksvm_classifier = SVC(kernel = 'rbf', C=1, gamma=0.6,random_state = 101)

ksvm_classifier.fit(X_train, y_train)
# Predicting the Test set results

ksvm_y_pred = ksvm_classifier.predict(X_test)
ksvm_classifier.fit(X_train_all, y_train_all)

ksvm_y_pred_all= ksvm_classifier.predict(X_test_all)
sub_ksvm = pd.DataFrame()

sub_ksvm['PassengerId'] = df_test['PassengerId']

sub_ksvm['Survived'] = ksvm_y_pred_all

sub_svm.to_csv('svm.csv',index=False)