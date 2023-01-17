!pip install fastai==0.7

!pip install numpy

!pip install scipy

!pip install seaborn

!pip install matplotlib

!pip install sklearn

!pip install pandas

!pip install xgboost



%load_ext autoreload

%autoreload 2



%matplotlib inline
import numpy as np

import scipy 

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn

import pandas as pd



from xgboost import XGBClassifier



from sklearn import ensemble, preprocessing, linear_model, model_selection, metrics 

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split



from pandas_summary import DataFrameSummary



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
df.head()
df.describe(include='all')
df.dtypes
df.shape
df.isnull().sum()
df = df.fillna(df.mean())

test_df = test_df.fillna(df.mean())

df.describe(include='all')
sns.countplot('Survived', data=df)

#print("Amount of people survived", df["Survived"][df["Survived"] == 1].value_counts(normalize = True)[1]*100)



print("Number of people who survived:", len(df[df['Survived'] > 0]))
# Establishing if there was a relationship between PassengerID and Survival



passenger_df = pd.cut(df['PassengerId'],

                      bins=[0, 99, 198, 297, 396, 495, 594, 693, 792, 891], 

                      labels=['0-99', '99-198', '198-297', '297-396', '396-495', '495-594', '594-693', '693-792', '792-891'])



sns.barplot('Survived', passenger_df, data=df)
#Establishing the relationship between Class and Survival



print("The number of people survived in Class 1: ", len(df[df['Pclass'] == 1]))

print("The number of people survived in Class 2: ", len(df[df['Pclass'] == 2]))

print("The number of people survived in Class 3: ", len(df[df['Pclass'] == 3]))

sns.barplot('Pclass', 'Survived', data=df)
#Establishing the relationship between Survival and Age



a = sns.FacetGrid(df, hue = 'Survived', aspect=4 )

a.map(sns.kdeplot, 'Age', shade= True)

a.set(xlim=(0 , df['Age'].max()))

a.add_legend()
df['Age'].describe()
age_df = pd.cut(df['Age'],

    bins=[0, 5, 12, 18, 25, 32, 40, 60, 80],

    labels=['Babies','Children','Teenagers', 'Young Adults', 'Adults', 'Middle-Age Adult', 'Senior', 'Older Senior']

)



plt.subplots(figsize=(10,5))

sns.set(style="ticks", color_codes=True)

sns.barplot(x=age_df, y=df['Survived'])

sns.pointplot(x=age_df, y=df['Survived'], color='black')
##Establishing a relationship between SibSp (Siblings/Spouses) and Survival

sns.barplot(x=df['SibSp'], y=df['Survived'])



print("The amount of people who survived with 0 siblings/spouses: ", 

      df['Survived'][df['SibSp'] == 0].value_counts(normalize=True)[1] * 100)

print("The amount of people who survived with 1 siblings/spouses: ", 

      df['Survived'][df['SibSp'] == 1].value_counts(normalize=True)[1] * 100)

print("The amount of people who survived with 2 siblings/spouses: ", 

      df['Survived'][df['SibSp'] == 2].value_counts(normalize=True)[1] * 100)

print("The amount of people who survived with 3 siblings/spouses: ", 

      df['Survived'][df['SibSp'] == 3].value_counts(normalize=True)[1] * 100)

print("The amount of people who survived with 4 siblings/spouses: ", 

      df['Survived'][df['SibSp'] == 4].value_counts(normalize=True)[1] * 100)
##Establishing a relationship between Parch (Parents/Children) and Survival

sns.barplot(x=df['Survived'], y=df['Parch'])
#Establishing a relationship between Fare fee and Survival

fare_df = pd.cut(df['Fare'],

                bins = [0, 25, 50, 75, 100, 200, 300, 400, 512],

                labels = ['0-25', '25-50', '50-75', '75-100', '100-200', '200-300', '300-400', '400+'])



plt.subplots(figsize=(10,5))

sns.barplot(x=fare_df, y=df['Survived'])
#Establishing a relationship between location embarked and Survival



sns.barplot(x=df['Embarked'], y=df['Survived'])
sns.heatmap(df.corr(), vmax=0.8)
##Pairplot of all the features

allPairPlot =  sns.pairplot(df, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )

allPairPlot.set(xticklabels=[])
#Dropping columns and making it suitable for predictions 

test_df = test_df.drop(['Name', 'Cabin', 'Ticket'], axis=1)

test_df['Sex'] = pd.Categorical(test_df.Sex)

test_df['Embarked'] = pd.Categorical(test_df.Embarked)



test_df['Sex'] = test_df.Sex.cat.codes

test_df['Embarked'] = test_df.Embarked.cat.codes



test_df.head()

features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
#Dropping the columns that are not needed and converting the ones that are needed into integers

dropped = df.drop(['Survived', 'Name', 'Cabin', 'Ticket'], axis=1)

target = df['Survived']



dropped['Sex'] = pd.Categorical(dropped.Sex)

dropped['Embarked'] = pd.Categorical(dropped.Embarked)

#dropped['PassengerId'] = pd.Categorical(dropped.PassengerId)



dropped['Sex'] = dropped.Sex.cat.codes 

#1 = Male, 0 = Female

dropped['Embarked'] = dropped.Embarked.cat.codes

#0 = Southampton, 1 = QueensTown, 2 = Cherborg

#dropped['PassengerId'] = dropped.PassengerId.cat.codes



X_train, X_val, y_train, y_val = train_test_split(dropped, target, test_size=0.25, random_state=0)

X_train.dtypes
#Creating an list that will handle accuracy scores

accuracies = []
#Logistic Regression



logistic = linear_model.LogisticRegression()

logistic.fit(X_train, y_train)

logisticscore = logistic.score(X_train, y_train)

y_pred = logistic.predict(X_val)

final_log = round(accuracy_score(y_pred, y_val) * 100, 2)

accuracies.append(final_log)

print(final_log)
#Gradient Boosting Classifier



gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)

gbc_score = gbc.score(X_train, y_train)

y_pred = gbc.predict(X_val)

final_gbc = round(accuracy_score(y_pred, y_val) * 100, 2)

accuracies.append(final_gbc)

print(final_gbc)
#Random Forest Classifier



rf_classifier = RandomForestClassifier(n_estimators = 20, criterion='entropy')

rf_classifier.fit(X_train, y_train)

rfc_score = rf_classifier.score(X_train, y_train)

y_pred = rf_classifier.predict(X_val)

final_rfc = round(accuracy_score(y_pred, y_val) * 100, 2)

accuracies.append(final_rfc)

print(final_rfc)
#SVC 



svc = SVC(kernel='linear')

svc.fit(X_train, y_train)

svc_score = svc.score(X_train, y_train)

y_pred = svc.predict(X_val)

final_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

accuracies.append(final_svc)

print(final_svc)
#KNN



knn = KNeighborsClassifier(p=2, n_neighbors = 10)

knn.fit(X_train, y_train)

knn_score = knn.score(X_train, y_train)

y_pred = knn.predict(X_val)

final_knn = round(accuracy_score(y_pred, y_val) * 100, 2)

accuracies.append(final_knn)

print(final_knn)
#XGBoost



xgboost = XGBClassifier()

xgboost.fit(X_train, y_train)

xgscores = xgboost.score(X_train, y_train)

y_pred = xgboost.predict(X_val)

final_xgb = round(accuracy_score(y_pred, y_val) * 100, 2)

accuracies.append(final_xgb)

print(final_xgb)
accuracies
accuracy_labels = ['Logistic Regression', 'Gradient Boosting Classifier', 'Random Forest Classifier', 'SVC', 'KNN', 'XGBoost']

sns.barplot(x=accuracies, y=accuracy_labels)
predictions = xgboost.predict(test_df[features])

predictions
submission = pd.DataFrame({'PassengerId' : test_df['PassengerId'], 'Survived': predictions})

submission.to_csv('submission.csv', index=False)
submission.tail()