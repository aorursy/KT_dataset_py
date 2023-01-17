import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

survey = pd.read_csv('../input/survey.csv')

survey.head()
print(survey.shape)

print(survey.dtypes)
survey.isnull().sum()
survey.drop(['Timestamp','comments'], axis=1, inplace=True)

print(survey.columns)
top_10_countries = survey['Country'].value_counts()[:10].to_frame()

plt.figure(figsize=(10,5))

sns.barplot(top_10_countries['Country'],top_10_countries.index,palette="PuBuGn_d")

plt.title('Top 10 Countries by number of respondents',fontsize=18,fontweight="bold")

plt.xlabel('')

plt.show()
usa = survey.loc[survey['Country'] == 'United States']

top_10_statesUS = usa['state'].value_counts()[:10].to_frame()

plt.figure(figsize=(10,5))

sns.barplot(top_10_statesUS['state'],top_10_statesUS.index,palette="PuBuGn_d")

plt.title('Top 10 US states contributing',fontsize=18,fontweight="bold")

plt.xlabel('')

plt.show()
def clean_age(age):

    if age>=0 and age<=100:

        return age

    else:

        return np.nan

survey['Age'] = survey['Age'].apply(clean_age)

plt.figure(figsize=(10,5))

sns.distplot(survey['Age'].dropna())

plt.title("Age Distribution",fontsize=20,fontweight="bold")

plt.show()
age = survey.loc[survey['treatment'] == 'Yes']

age['Age'].value_counts().nlargest(1)

print("32 years is the age in which person take mental treatment.")
survey['Gender'] = survey['Gender'].str.lower()

male = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "cis male"]

trans = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]

female = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

survey['Gender'] = survey['Gender'].apply(lambda x:"Male" if x in male else x)

survey['Gender'] = survey['Gender'].apply(lambda x:"Female" if x in female else x)

survey['Gender'] = survey['Gender'].apply(lambda x:"Trans" if x in trans else x)

survey.drop(survey[survey.Gender == 'p'].index, inplace=True)

survey.drop(survey[survey.Gender == 'a little about you'].index, inplace=True)
plt.figure(figsize=(10,5))

sns.countplot('Gender',data = survey, order = survey['Gender'].value_counts().index,palette="PuBuGn_d")

plt.title("Gender Counts",fontsize=20,fontweight="bold")

plt.show()
plt.figure(figsize=(6,5))

sns.countplot(survey['treatment'],palette="PuBuGn_d")

plt.title("Treatment Distribution",fontsize=18,fontweight="bold")

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(y="Gender", hue="treatment", data=survey)

plt.title("Mental health wrt Gender",fontsize=18,fontweight="bold")

plt.ylabel("")

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(y="family_history", hue="treatment", data=survey)

plt.title("Does family hisitory effects mental health ? ",fontsize=18,fontweight="bold")

plt.ylabel("")

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(y="work_interfere", hue="treatment", data=survey)

plt.title("Treatment depends on work interfere ?",fontsize=18,fontweight="bold")

plt.ylabel("")

plt.show()
plt.figure(figsize=(10,5))

sns.countplot('no_employees',data = survey, order = survey['no_employees'].value_counts().index,palette="PuBuGn_d")

plt.title("Employee Count of Companies",fontsize=20,fontweight="bold")

plt.show()
plt.figure(figsize=(10,5))

sns.countplot("no_employees", hue="treatment", data=survey)

plt.title("Employee count vs Treatment",fontsize=18,fontweight="bold")

plt.ylabel("")

plt.show()
from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

for i in survey.columns:

    survey[i] = number.fit_transform(survey[i].astype('str'))
corr=survey.corr()['treatment']

corr[np.argsort(corr,axis=0)[::-1]]
features_correlation = survey.corr()

plt.figure(figsize=(8,8))

sns.heatmap(features_correlation,vmax=1,square=True,annot=False,cmap='Blues')

plt.show()
from sklearn.model_selection import train_test_split

X = survey.drop(['treatment','benefits','wellness_program','seek_help','anonymity','supervisor'], axis=1)

y = survey.treatment

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1234)
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import accuracy_score



xgb = XGBClassifier(learning_rate =0.01,

 n_estimators=5000,

 max_depth=4,

 min_child_weight=6,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 reg_alpha=0.005,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)



xgb.fit(X_train,y_train)

y_pred = xgb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print('ACCURACY : ',accuracy*100,'%')
features = X.columns

for name, importance in zip(features, xgb.feature_importances_):

    print(name, "=", importance)



importances = xgb.feature_importances_

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), features) 

plt.xlabel('Relative Importance')

plt.show()
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test,y_pred)

sns.heatmap(confusion_matrix,annot=True,fmt='',cmap='Blues')

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()