import pandas as pd 

import seaborn as sns

from pandas_profiling import ProfileReport
import numpy as np 
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test  = pd.read_csv('/kaggle/input/titanic/test.csv')
#Some algorithms like XGBOOST and LightGBM can treat missing data without any preprocessing 
train.head()
train.info()
test.info()
import sweetviz

my_report = sweetviz.compare([train, "Train"], [test, "Test"], "Survived")
my_report.show_html("Report.html")
print('Shape of TRAIN DATA : ', train.shape)

print('Shape of TEST DATA : ', test.shape)
#Examining the target column 

train['Survived'].value_counts()
s = sns.countplot(x = 'Survived', data = train)

# 0 - Did not Survive

# 1 - Survived
def missing_values_table(df):

    mis_val = df.isnull().sum()

    mis_val_percent = 100 * df.isnull().sum() / len(df)

    mis_val_table = pd.concat([mis_val,mis_val_percent],axis=1) 

    mis_val_table_rename = mis_val_table.rename(columns = {0:'Missing Values',

                                                          1:'% of Total Missing Values'})

    mis_val_table_rename_sort = mis_val_table_rename[mis_val_table_rename.iloc[:,1] !=0].sort_values('% of Total Missing Values',ascending = False).round(1)

    print('Your selected dataframe has ' + str(df.shape[1]) + ' columns')

    print('There are ' + str(mis_val_table_rename_sort.shape[0]) + ' columns have missing values')

    return mis_val_table_rename_sort

    
train_missing = missing_values_table(train)

train_missing
test_missing = missing_values_table(test)

test_missing
import missingno as msno
msno.bar(train)
msno.bar(test)
msno.matrix(train)
msno.matrix(test)
msno.matrix(train)
sorted = train.sort_values('Age')

msno.matrix(sorted)
sorted1 = train.sort_values('Cabin')

msno.matrix(sorted1)
msno.heatmap(train)
msno.dendrogram(train)
train.isnull().sum()
train_1 = train.copy()

Age_mean= train_1['Age'].mean()

Age_mean
train_1['Age'] = train_1['Age'].fillna(value=Age_mean)
msno.matrix(train_1)
from sklearn.impute import SimpleImputer

train_mf = train.copy()

mean_imputer = SimpleImputer(strategy='mean')

median_imputer = SimpleImputer(strategy='median')

most_freq_imputer = SimpleImputer(strategy='most_frequent')

constant_imputer = SimpleImputer(strategy='constant')

train_mf.iloc[:,:] = most_freq_imputer.fit_transform(train_mf)

msno.matrix(train_mf)
test.head()
test.info()
test.shape
test_missing = missing_values_table(test)

test_missing
msno.bar(test)
msno.matrix(test)
msno.dendrogram(test)
msno.heatmap(test)
# CABIN HAS 77% MISSING VALUES IN TRAIN

# CABIN HAS 78% MISSING VALUES IN TEST 

# THEREFORE DROPPING CABIN COLUMN WOULD BE BETTER FOR ANALYSIS
test = test.drop(columns=['Cabin'])

train_mf = train_mf.drop(columns=['Cabin'])
missing_values_table(test)
from sklearn.impute import SimpleImputer

test_mf = test.copy()

mean_imputer = SimpleImputer(strategy='most_frequent')

test_mf.iloc[:,:] = mean_imputer.fit_transform(test_mf)

msno.matrix(test_mf)
#Now both the training and testing data are cleaned
profile_train = ProfileReport(train_mf)

profile_train
profile_test = ProfileReport(test_mf)

profile_test
import seaborn as sns 

import matplotlib.pyplot as plt
train_mf.columns
sns.barplot(x='Sex',y='Survived',data=train_mf)

plt.xlabel('Gender of the Passenger')

plt.ylabel('Proportion survived')

plt.title('Survival Based on Gender')
female_survived = train_mf[train_mf['Sex']=='female']['Survived'].sum()

male_survived   = train_mf[train_mf['Sex']=='male']['Survived'].sum()

Class_1         = train_mf[train_mf['Pclass']==1]['Survived'].sum()

Class_2         = train_mf[train_mf['Pclass']==2]['Survived'].sum()

Class_3         = train_mf[train_mf['Pclass']==3]['Survived'].sum()



print('No. of females survived : ' , female_survived)

print('No. of male survived : ' , male_survived)

print('No. of Class 1 passengers survived : ' , Class_1)

print('No. of Class 2 passengers survived : ' , Class_2)

print('No. of Class 3 passengers survived : ' , Class_3)

print('Total no. of Survivors = ' , female_survived+male_survived)
sns.barplot(x='Pclass',y='Survived',data=train_mf)

plt.xlabel('Class of Passenger')

plt.ylabel('Proportion Survived')

plt.title('Survival Based on Passenger Class')
sns.barplot(x='Pclass',y='Survived',hue='Sex',data=train_mf)

plt.title('Survival Based on Gender and Passenger Class')
survived_ages = train_mf[train_mf['Survived']==1]['Age']

not_survived_ages  = train_mf[train_mf['Survived']==0]['Age']

plt.subplot(1,2,1)

sns.distplot(survived_ages,kde=False)

plt.ylabel('Proportion')

plt.title('Survived')

plt.subplot(1,2,2)

sns.distplot(not_survived_ages,kde=False)

plt.ylabel('Proportion')

plt.title('Died')

plt.show()
fare_survived = train_mf[train_mf['Survived']==1]['Fare']

fare_died     = train_mf[train_mf['Survived']==0]['Fare']

plt.subplot(1,2,1)

sns.distplot(fare_survived,kde=False)

plt.ylabel('Fare')

plt.title('Fare of Survived')

plt.subplot(1,2,2)

sns.distplot(fare_died,kde=False)

plt.ylabel('Fare')

plt.title('Fare of Died')

plt.show()
sns.stripplot(x='Survived',y='Age',data=train_mf,jitter=True)

plt.title('Ages vs Survival')
sns.pairplot(train_mf)
train_mf.sample(5)
test_mf.sample(5)
# Changing Male = 1 and Female = 0

def label_encode(sex):

    if sex == 'male':

        label = 1

    else:

        label = 0

    return label



train_mf['Sex']=train_mf['Sex'].apply(label_encode)

def label_encode(sex):

    if sex == 'male':

        label = 1

    else:

        label = 0

    return label

test_mf['Sex']=test_mf['Sex'].apply(label_encode)
train_mf.head()
train_mf['FamSize'] = train_mf['SibSp'] + train_mf['Parch'] + 1 

test_mf['FamSize'] = test_mf['SibSp'] + test_mf['Parch'] + 1
train_mf.head()
test_mf.head()
train_mf['IsAlone'] = train_mf.FamSize.apply(lambda x: 1 if x==1 else 0)

test_mf['IsAlone'] = test_mf.FamSize.apply(lambda x: 1 if x==1 else 0)
train_mf.sample(5)
test_mf.sample(5)
from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.metrics import make_scorer,accuracy_score
from sklearn.model_selection import GridSearchCV
def logic_encoder(embark):

    if embark == 'S':

        logic = 0

    elif embark == 'C':

        logic = 1

    else:

        logic = 2

    return logic

train_mf['Embarked'] = train_mf['Embarked'].apply(logic_encoder)
def logic_encoder(embark):

    if embark == 'S':

        logic = 0

    elif embark == 'C':

        logic = 1

    else:

        logic = 2

    return logic

test_mf['Embarked'] = test_mf['Embarked'].apply(logic_encoder)
test_mf['Name'] = test_mf['Name'].astype('object')

train_mf['Name'] = train_mf['Name'].astype('object')
features = ['Pclass','Sex', 'Age', 'SibSp','Parch','Fare','Embarked', 'FamSize', 'IsAlone']

X_train = train_mf[features]

y_train = train_mf['Survived']

X_test  = test_mf[features]
from sklearn.model_selection import train_test_split
X_training,X_valid,y_training,y_valid = train_test_split(X_train,y_train,test_size=0.2,random_state=0)
svc_clf = SVC() 

svc_clf.fit(X_training, y_training)

pred_svc = svc_clf.predict(X_valid)

acc_svc = accuracy_score(y_valid, pred_svc)



print(acc_svc)
linsvc_clf = LinearSVC()

linsvc_clf.fit(X_training, y_training)

pred_linsvc = linsvc_clf.predict(X_valid)

acc_linsvc = accuracy_score(y_valid, pred_linsvc)



print(acc_linsvc)
rf_clf = RandomForestClassifier()

rf_clf.fit(X_training, y_training)

pred_rf = rf_clf.predict(X_valid)

acc_rf = accuracy_score(y_valid, pred_rf)



print(acc_rf)
logreg_clf = LogisticRegression()

logreg_clf.fit(X_training, y_training)

pred_logreg = logreg_clf.predict(X_valid)

acc_logreg = accuracy_score(y_valid, pred_logreg)



print(acc_logreg)
knn_clf = KNeighborsClassifier()

knn_clf.fit(X_training, y_training)

pred_knn = knn_clf.predict(X_valid)

acc_knn = accuracy_score(y_valid, pred_knn)



print(acc_knn)
gnb_clf = GaussianNB()

gnb_clf.fit(X_training, y_training)

pred_gnb = gnb_clf.predict(X_valid)

acc_gnb = accuracy_score(y_valid, pred_gnb)



print(acc_gnb)
dt_clf = DecisionTreeClassifier()

dt_clf.fit(X_training, y_training)

pred_dt = dt_clf.predict(X_valid)

acc_dt = accuracy_score(y_valid, pred_dt)



print(acc_dt)
xg_clf = XGBClassifier(objective="binary:logistic", n_estimators=10, seed=123)

xg_clf.fit(X_training, y_training)

pred_xg = xg_clf.predict(X_valid)

acc_xg = accuracy_score(y_valid, pred_xg)



print(acc_xg)
model_performance = pd.DataFrame({"Model": ["SVC", "Linear SVC", "Random Forest", "Logistic Regression", "K Nearest Neighbors", "Gaussian Naive Bayes","Decision Tree", "XGBClassifier"],

                                  "Accuracy": [acc_svc, acc_linsvc, acc_rf, acc_logreg, acc_knn, acc_gnb, acc_dt, acc_xg]})



model_performance.sort_values(by="Accuracy", ascending=False)
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score



skf = StratifiedKFold(n_splits=10,random_state=23)

score = cross_val_score(rf_clf,X_training, y_training,cv = skf)

print(f'Accuracy Score After Cross Validation: {np.mean(score)*100:4.1f}%')
#accuracy

score = cross_val_score(rf_clf,X_training, y_training, cv=skf)

#positive precision

precision_score = cross_val_score(rf_clf,X_training, y_training, cv=skf, scoring='precision')

#positive recall

recall_score = cross_val_score(rf_clf,X_training, y_training, cv=skf, scoring='recall')

#auc

auc_score = cross_val_score(rf_clf,X_training, y_training, cv=skf, scoring='roc_auc')



print(f'Accuracy Score from Cross Validation: {np.mean(score)*100:4.1f}%')

print(f'Positive Precision Rate from Cross Validation: {np.mean(precision_score)*100:4.1f}%')

print(f'Positive Recall Rate from Cross Validation: {np.mean(recall_score)*100:4.1f}%')

print(f'Area Under Curve from Cross Validation: {np.mean(auc_score)*100:4.1f}%')