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
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
import sklearn
%matplotlib inline
sns.set(style = "whitegrid")
from scipy import stats
#Import the titanitc training dataset

df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head(10)
df.info()
df.shape
#Survival distribution count

df['Survived'].value_counts()
df.describe()

#Below we can see that the mean of the age distribution in the boat is around 29 years which also means that many of the passengers are pretty much young adults.
#mean of the passenger fare is around $32 with max fare amount of $512 which is interesting.
#Below data shows that Age column has 177 nulls , cabin type as 687 nulls 

df.isnull().sum()


#To check the 2 missing values 

df['Embarked'].value_counts()
categorical = [cat for cat in df.columns if df[cat].dtype == 'O']

print('The categorical variables are', categorical)
df[categorical]
numerical = [num for num in df.columns if df[num].dtype == 'I']

print('The numerical variables are', numerical)
df['Sex'].value_counts()
#This shows that most of them are not accompanied by their parents or children

df.Parch.value_counts()
#This shows that most of them are not accompanied by their siblings or spouses

df.SibSp.value_counts()
#visualize the survival percentage vs age distribution on the ship

f,ax = plt.subplots(1,2,figsize=(18,8))

ax[0] = df['Survived'].value_counts().plot.pie(autopct='%2.2f%%',ax=ax[0],shadow=False)
ax[0].set_title('Survived % Dist')

ax[1] = sns.countplot(x = "Survived", data=df, palette="Set2", hue = 'Sex')
ax[1].set_title('Sex Dist')
plt.show()

#A very interesting thing to observe is that the overall survival is only 38%  and out of those majority of the survivors are females and a big majority of people who lost lives are males.
#visualize the siblings accompanied percentage vs survival distribution on the ship

f,ax = plt.subplots(figsize=(18,8))

ax = sns.countplot(x = "SibSp", data=df, palette="Set1", hue = 'Survived')
ax.set_title('SibSp % Dist')
plt.show()

#Very interesting thing to see is that out of the survivors group the ones who are single most likely did not survive vs the ones accompanied by a spouse/sibling did survive
#visualize the Parents/children accompanied percentage vs survival distribution on the ship

f,ax = plt.subplots(figsize=(18,8))
ax = sns.countplot(x = "Parch", data=df, palette="Set3", hue = 'Survived')
ax.set_title('SibSp Dist')
plt.show()

#Its observed that out of the survivors group the ones who are single most likely did not survive vs the ones accompanied by parents/children did survive
#visualize the survival percentage vs passenger class distribution on the ship

f,ax = plt.subplots(figsize=(18,8))

ax = df['Pclass'].value_counts().plot.pie(autopct='%2.2f%%',shadow=False)
ax.set_title('Pclass % Dist')

#Majority of the passengers are from lower class
#visualize the survival vs passenger class distribution on the ship

f,ax = plt.subplots(figsize=(8,8))

ax = sns.countplot(x = "Pclass", data=df, palette="Set1", hue = 'Survived')
ax.set_title('Passenger class % Dist')
plt.show()

#Its been observed that the majority of survivors are from first class followed by second class whereas the third class passengers are the least survived.
#distribution of age

f, ax = plt.subplots(figsize=(10,8))
ax = sns.distplot(df['Age'],bins=10,color = 'orange')
ax.set_title("Distribution of age variable")
#ax.set_xticklabels(df.age.value_counts().index, rotation = 30)
plt.show()

#Majority of the age group is in 20s and 30s
f, ax = plt.subplots(figsize=(10,8))
ax = sns.boxplot(x = 'Survived',y = 'Age', data = df)
ax.set_title("Outliers is Age variable")
plt.show()

##There are outliers in the age distribution in both survival vs non survival groups
#Passenger class vs age 

f, ax = plt.subplots(figsize=(10,8))
ax = sns.boxplot(x = 'Pclass',y = 'Age', hue = 'Survived', data = df)
ax.set_title("Outliers is Age variable")
plt.show()

#There are more outliters in the age in third class passengers group
df.corr()
df.corr().style.format("{:.5}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
#visualize the survival vs fare on the ship

f, ax = plt.subplots(figsize=(10,8))
ax = sns.barplot(x = 'Survived',y = 'Fare', data = df)
ax.set_title("Outliers is Fare variable")
plt.show()

##Its observed that passengers who brought cheaper tickets are more likely not to survive
#sns.pairplot(df)
#plt.show()
cf = df
cf.shape
df['Age'].isnull().sum()
cf['Age'].isnull().sum()
df['Age'].fillna(df['Age'].mode()[0], inplace=True)
#df1['Age'].isnull().sum()
#distribution of age

f, ax = plt.subplots(figsize=(10,8))
ax = sns.distplot(df['Age'],bins=10,color = 'orange')
ax.set_title("Distribution of age variable")
#ax.set_xticklabels(df.age.value_counts().index, rotation = 30)
plt.show()

#Majority of the age group is in 20s and 30s

f, ax = plt.subplots(figsize=(10,8))
ax = sns.distplot(cf['Age'],bins=10,color = 'orange')
ax.set_title("Distribution of age variable")
#ax.set_xticklabels(df.age.value_counts().index, rotation = 30)
plt.show()
df[['Survived','Pclass','Sex','Age','SibSp','Parch','Embarked','Fare']]
#Remove Nulls

df = df[df['Embarked'].notna()]
#Below are the columns that we will consider to train the data


X = df[['Pclass','Sex','Age','SibSp','Parch','Embarked','Fare']]


#Below are the columns that we will consider to train the data

y = df['Survived']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)
X_train.isnull().sum()
X_train.shape,y_train.shape
y_train.isnull().sum()
#Now lets import the testing dataset 

ef = pd.read_csv('/kaggle/input/titanic/test.csv')
ef.shape
ef.isnull().sum()
ef['Age'].fillna(ef['Age'].mode()[0], inplace=True)
ef['Fare'].fillna(ef['Fare'].mode()[0], inplace=True)
ef.isnull().sum()
#Below are the columns that we will consider to test the data


train = ef[['Pclass','Sex','Age','SibSp','Parch','Embarked','Fare']]
train.head(10)
categorical
# import category encoders

import category_encoders as ce

encoder = ce.OneHotEncoder(cols=['Sex','Embarked'])

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
X_train.head()
X_test.shape
X_test.head()
import sklearn

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state = 0)

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

# Check accuracy score 

from sklearn.metrics import accuracy_score

print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# instantiate the classifier with n_estimators = 100

rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)


rfc_100.fit(X_train, y_train)


y_pred_100 = rfc_100.predict(X_test)

print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))
#view the feature scores

feature_scores = pd.Series(rfc_100.feature_importances_,index = X_train.columns).sort_values(ascending=False)

feature_scores
#Visualzie the feature scores

f, ax = plt.subplots(figsize=(8,5))
ax = sns.barplot(x = feature_scores,y=feature_scores.index, data=df, color="Blue")
ax.set_title("Feature Importance Scores")
plt.show()
# Print the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)
from sklearn.metrics import classification_report

cr = classification_report(y_test,y_pred)

print('Classification Report \n', cr)
#Logistic Regression

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()


log.fit(X_train, y_train)


y_pred_log = log.predict(X_test)

print('Model accuracy score with Log Reg : {0:0.4f}'. format(accuracy_score(y_test, y_pred_log)))
#Bagging Classifier

from sklearn.ensemble import BaggingClassifier

from sklearn import tree

Bag = BaggingClassifier(random_state=0)

Bag.fit(X_train, y_train)

y_pred_bag = Bag.predict(X_test)

print('Model accuracy score with Bagging Classifier : {0:0.4f}'. format(accuracy_score(y_test, y_pred_bag)))
#XGboost

import xgboost as xgb #base_estimator = rfc,

xgbd=xgb.XGBClassifier(random_state=1,learning_rate=0.1)

xgbd.fit(X_train, y_train)

y_pred_xgbd = xgbd.predict(X_test)

print('Model accuracy score with Bagging Classifier : {0:0.4f}'. format(accuracy_score(y_test, y_pred_xgbd)))
#GaussianNB
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred_gauss = gaussian.predict(X_test)
gaussian_accy = round(accuracy_score(y_pred, y_test), 3)
print('Model accuracy score with GaussianNB : {0:0.4f}'. format(accuracy_score(y_test, y_pred_gauss)))
df['Cabin'].isnull().sum()
ef['Cabin'].isnull().sum()
#Trying to make use of Cabin feature.

#Replace nulls with N
df.Cabin.fillna("N", inplace=True)

#Grab the initial letter from column Cabin since the numbers after the letter don't really matter except the letter as it signifies the cabin class.
df.Cabin = [i[0] for i in df.Cabin]

#Replace nulls with N
ef.Cabin.fillna("N", inplace=True)

#Grab the initial letter from column Cabin since the numbers after the letter don't really matter except the letter as it signifies the cabin class.
ef.Cabin = [i[0] for i in ef.Cabin]

df.Cabin.value_counts()
#Based on the mean Fare price for each cabin type we can fill the blank Cabins with appropriate letters by considering the fare of the individual passengers

df.groupby(df.Cabin)['Fare'].mean().sort_values()
Cabin_N_train = df[df.Cabin == 'N']

Cabin_NoN_train = df[df.Cabin != 'N']

Cabin_N_test = ef[ef.Cabin == 'N']

Cabin_NoN_test = ef[ef.Cabin != 'N']
#For training Data

def cabin_est_tr(n):
    a = 0
    if n <= 15:
        a = 'G'
    if n >= 16 and n <= 26:
        a = 'F'
    if n >= 27 and n <= 37:
        a = 'T'
    if n >= 38 and n <= 42:
        a = 'A'
    if n >= 43 and n <= 52:
        a = 'E'
    if n >= 53 and n <= 80:
        a = 'D'    
    if n >= 81 and n <= 107:
        a = 'C'    
    else:
        a = 'B'
    return a 
        
ef.groupby(ef.Cabin)['Fare'].mean().sort_values()
#For testing Data

def cabin_est_te(n):
    a = 0
    if n <= 16.5:
        a = 'G'
    if n > 16.5 and n <= 30:
        a = 'F'
    if n >= 31 and n <= 43.5:
        a = 'D'
    if n >= 43.6 and n <= 64:
        a = 'A'
    if n >= 65 and n <= 102:
        a = 'E'
    if n >= 103 and n <= 133:
        a = 'C'       
    else:
        a = 'B'
    return a 
Cabin_N_train['Cabin'] = Cabin_N_train.Fare.apply(lambda x: cabin_est_tr(x))
Cabin_N_test['Cabin'] = Cabin_N_test.Fare.apply(lambda y: cabin_est_te(y))
## getting the modified cabin data back to the original datasets 
df = pd.concat([Cabin_N_train, Cabin_NoN_train], axis=0)

ef = pd.concat([Cabin_N_test, Cabin_NoN_test], axis=0)
df.shape
df.head(2)
ef.shape
#Feature Engineering

#Creating a new column called name length that calculates the length of the name.

df['Name_len'] = [len(i) for i in df.Name]
ef['Name_len'] = [len(i) for i in ef.Name]

def name_length(size):
    a = ''
    if size <= 20:
        a = 'Short'
    elif size <= 30:
        a = 'Medium'
    elif size <= 40:
        a = 'Large'    
    else:
        a = 'Very large'
    return a            
df['Name_size_cat'] = df.Name_len.apply(lambda x: name_length(x))

ef['Name_size_cat'] = ef.Name_len.map(name_length)
ef.head(10)
#Create a new feature by adding the total number of people accompanied

df['total_members'] = df['SibSp']+df['Parch']+1 #including the passengerid
ef['total_members'] = ef['SibSp']+ef['Parch']+1 #including the passengerid

#Create a new feature to check if a passenger is accompanied by family or alone

df['Family_In'] = [0 if i == 1 else 1 for i in df.total_members]

ef['Family_In'] = [0 if i == 1 else 1 for i in ef.total_members]
df.head(10)
ef.head(10)
#Create a new feature to calculate the fare per passenger

df['Fare'] = round((df['Fare']),1)

ef['Fare'] = round((ef['Fare']),1)

df['C_Fare'] = round((df['Fare']/df['total_members']),1)

ef['C_Fare'] = round((ef.Fare/ef.total_members),1)

df.head(10)
#Create the fare group

def fare_group(fare):
    a = ''
    if fare <= 10:
        a = 'Very low'
    elif fare <= 20:
        a = 'Low'
    elif fare <= 40:
        a = 'Medium'
    elif fare <= 60:
        a = 'High'    
    else:
        a = 'Very High'
    return a        

df['FareGp'] = df['C_Fare'].map(fare_group)

ef['FareGp'] = ef['C_Fare'].map(fare_group)
ef.head(20)
#Below are the columns that we will consider to train the data

X = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked','Name_len','Name_size_cat','total_members','Family_In','C_Fare','FareGp']]

#Below are the columns that we will consider to train the data

y = df['Survived']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 0)

# import category encoders

import category_encoders as ce

encoder = ce.OneHotEncoder(cols=['Sex','Cabin','Embarked','Name_size_cat','FareGp'])

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)


X_train.shape,X_test.shape
headers = X_train.columns

X_train.head()
#Feature Scaling to normalize the data to avoid the columns with bigger numbers which can affect the model performance

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)
#After Scaling

pd.DataFrame(X_train,columns = headers).head()
pd.DataFrame(X_test,columns = headers).head()
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)
rfc_100.fit(X_train, y_train)
y_pred_100 = rfc_100.predict(X_test)
print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))
#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
#Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print('Classification Report \n', cr)
#Logistic Regression

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train, y_train)
y_pred_log = log.predict(X_test)
print('Model accuracy score with Log Reg : {0:0.4f}'. format(accuracy_score(y_test, y_pred_log)))
#Bagging Classifier
from sklearn.ensemble import BaggingClassifier
BaggingClassifier = BaggingClassifier(random_state=0)
BaggingClassifier.fit(X_train, y_train)
y_pred_bagging = BaggingClassifier.predict(X_test)
print('Model accuracy score with Bagging Classifier : {0:0.4f}'. format(accuracy_score(y_test, y_pred_bagging)))
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
GradientBoostingClassifier = GradientBoostingClassifier()
GradientBoostingClassifier.fit(X_train, y_train)
y_pred_GBClassifier = svc.predict(X_test)
print('Model accuracy score with GBClassifier : {0:0.4f}'. format(accuracy_score(y_test, y_pred_GBClassifier)))
#XGBoost Classifier
import xgboost as xgb #base_estimator = rfc,
xgbd=xgb.XGBClassifier(random_state=1,learning_rate=0.1)
xgbd.fit(X_train, y_train)
y_pred_xgbd = xgbd.predict(X_test)
print('Model accuracy score with XGBoost Classifier : {0:0.4f}'. format(accuracy_score(y_test, y_pred_xgbd)))
# Print the Confusion Matrix and slice it into four pieces for bagging classifier

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_xgbd)

print('Confusion matrix\n\n', cm)
#Classification Report for bagging classifier

from sklearn.metrics import classification_report

cr = classification_report(y_test,y_pred_xgbd)

print('Classification Report \n', cr)
from sklearn.ensemble import ExtraTreesClassifier
ExtraTreesClassifier = ExtraTreesClassifier()
ExtraTreesClassifier.fit(X_train, y_train)
y_pred_extra = ExtraTreesClassifier.predict(X_test)
print('Model accuracy score with ExtraTreesClassifier : {0:0.4f}'. format(accuracy_score(y_test, y_pred_extra)))
from sklearn.gaussian_process import GaussianProcessClassifier
GaussianProcessClassifier = GaussianProcessClassifier()
GaussianProcessClassifier.fit(X_train, y_train)
y_pred_gausspclassifier = GaussianProcessClassifier.predict(X_test)
print('Model accuracy score with GaussianProcessClassifier : {0:0.4f}'. format(accuracy_score(y_test, y_pred_gausspclassifier)))
from sklearn.tree import DecisionTreeClassifier
DecisionTreeClassifier = DecisionTreeClassifier(max_depth = 6,random_state = 0)
DecisionTreeClassifier.fit(X_train, y_train)
y_pred_DTclassifier = DecisionTreeClassifier.predict(X_test)
print('Model accuracy score with DecisionTreeClassifier : {0:0.4f}'. format(accuracy_score(y_test, y_pred_DTclassifier)))
from sklearn.svm import SVC 
svc = SVC(gamma = 0.1)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print('Model accuracy score with SVC : {0:0.4f}'. format(accuracy_score(y_test, y_pred_svc)))
#Voting Classifier - voting hard

from sklearn.ensemble import VotingClassifier 
Voting_hard = VotingClassifier(estimators=[
    ('RandomForestClassifier', rfc_100),
    ('LogisticRegression', log),
    ('BaggingClassifier',BaggingClassifier),
    ('GradientBoostingClassifier', GradientBoostingClassifier),
    ('XGBClassifier', xgbd),
    ('ExtraTreesClassifier', ExtraTreesClassifier),
    ('GaussianProcessClassifier',GaussianProcessClassifier),
    ('DecisionTreeClassifier', DecisionTreeClassifier),
    ('svc', svc)
],voting='hard')
Voting_hard.fit(X_train, y_train)
y_pred_vot_hard = Voting_hard.predict(X_test) 
print('Model accuracy score with voting hard : {0:0.4f}'. format(accuracy_score(y_test, y_pred_vot_hard)))
# Print the Confusion Matrix and slice it into four pieces for VotingClassifier

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_vot_hard)

print('Confusion matrix\n\n', cm)

#Classification Report for VotingClassifier

from sklearn.metrics import classification_report

cr = classification_report(y_test,y_pred_vot_hard)

print('Classification Report \n', cr)
#Now lets test our final model on the actual testing dataset to submit the results

ef.head(30) #Test dataset
#Below are the columns that we will consider for our final test data

X_test_final = ef[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked','Name_len','Name_size_cat','total_members','Family_In','C_Fare','FareGp']]

import category_encoders as ce

encoder = ce.OneHotEncoder(cols=['Sex','Cabin','Embarked','Name_size_cat','FareGp'])

X_test_final = encoder.fit_transform(X_test_final)
#Scaling final test data

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_test_final = scaler.fit_transform(X_test_final) 
#After Scaling final test data

pd.DataFrame(X_test_final,columns = headers).head()
#Predicting the survival for the final test dataset

y_pred_final_test = Voting_hard.predict(X_test_final)
print('Model results for the prediction of the titanic survival',y_pred_final_test)
finalsubmission = ef[['PassengerId']].copy()
Survived=pd.DataFrame(y_pred_final_test, columns=['Survived'])
submission = finalsubmission.join(Survived)
submission['Survived'].value_counts()
submission.to_csv("titanic_submission.csv", index=False)