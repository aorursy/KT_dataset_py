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
import pandas as pd

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.info()
df.describe()
df.Education.unique()
df.isnull().any()
df.isnull().sum()


Education = {1: "Below College", 2: "College", 3:"Bachelor", 4:"Master", 5:"Doctor"}

EnvironmentSatisfaction = {1: "Low", 2: "Medium", 3:"High", 4:"Very High"}

JobInvolvement = {1: "Low", 2: "Medium", 3:"High", 4:"Very High"}

JobSatisfaction = {1: "Low", 2: "Medium", 3:"High", 4:"Very High"}

PerformanceRating = {1: "Low", 2: "Good", 3:"Excellent", 4:"Outstanding"}

RelationshipSatisfaction = {1: "Low", 2: "Medium", 3:"High", 4:"Very High"}

WorkLifeBalance = {1: "Bad", 2: "Good", 3:"Better", 4:"Best"}



df.replace({"Education": Education, "JobInvolvement":JobInvolvement, "JobSatisfaction": JobSatisfaction, 

              "PerformanceRating":PerformanceRating, "RelationshipSatisfaction":RelationshipSatisfaction,

             "WorkLifeBalance":WorkLifeBalance, "EnvironmentSatisfaction":EnvironmentSatisfaction}, inplace=True)
df.head()
import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('seaborn')
sns.countplot(df.Attrition,hue=df.Department,palette='muted')
sns.countplot(df.Attrition,hue=df.Education,palette='muted')
sns.countplot(df.Attrition,hue=df.EducationField,palette='muted')
sns.countplot(df.Gender,hue=df.JobSatisfaction,palette='muted')
sns.countplot(df.Gender,hue=df.OverTime,palette='muted')
sns.lineplot(df.Age,df.HourlyRate,color='#C70039')
sns.lineplot(df.YearsAtCompany,df.YearsInCurrentRole,color='#FF5733')
cat_cols = [x for x in df.columns if df[x].dtype == 'object']

cat_data = df[cat_cols]

num_cols = df.columns.difference(cat_cols)

num_data = df[num_cols]
corr = num_data.corr()

heatmap = sns.heatmap(corr, vmin=0, vmax=1)
#Categorical Data

cat_data = pd.concat([cat_data.reset_index(drop=True)],axis=1)

cat_data.columns



fig, axarr = plt.subplots(3, 2, figsize=(18, 15))

sns.countplot(x="BusinessTravel", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[0][0],palette="muted") 

# Higher attrition in travel Frequently

sns.countplot(x="OverTime", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[0][1],palette="muted")

# Higher attrition in Overtime

sns.countplot(x="Education", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[1][0],palette="muted")

sns.countplot(x="EducationField", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[1][1],palette="muted")

# Higher percentage attrition in merketing and tech degree

sns.countplot(x="JobInvolvement", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[2][0],palette="muted")

sns.countplot(x="JobSatisfaction", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[2][1],palette="muted")

# Extremely low attrition in Very High Job satisfaction

# higher attrition in "High" vs ["Very High and "Medium"]



fig, axarr = plt.subplots(2, 2, figsize=(15, 10))

sns.countplot(x="Gender", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[0][0],palette="muted")

sns.countplot(x="MaritalStatus", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[0][1],palette="muted")

# More attrition among single people

sns.countplot(x="RelationshipSatisfaction", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[1][0],palette="muted")

# More attrition among people with low relationship satisfaction

sns.countplot(x="WorkLifeBalance", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[1][1],palette="muted")

# High attrition among people with bad Work Life Balance, even though very few people have reported bad worklife balance



fig, axarr = plt.subplots(2, figsize=(19, 10))

# Higher percentage attrition in Sales

sns.countplot(x="Department", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[0],palette="muted")

# Higher attrition in Sales roles and Lab technician role

sns.countplot(x="JobRole", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[1],palette="muted")



# Overtime vs Department

sns.factorplot(x="OverTime", col="Department", col_wrap=4,hue="Attrition",

                   data=cat_data, kind ="count",palette="muted")



# JobInvolvement vs JobSatisfaction

sns.factorplot(x="JobInvolvement", col="JobSatisfaction", col_wrap=4,hue="Attrition",

                   data=cat_data, kind ="count",palette="muted")
X = num_data

df['Attrition'] = df['Attrition'].replace(['Yes','No'],['1','0']).astype('int64')



y = df.Attrition
#Logistic Regression with R-square

import statsmodels.api as sm

logit_model=sm.Logit(y,X)

result=logit_model.fit()

print(result.summary2())
#Class Imbalance SMOTE

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', random_state=100, k_neighbors=5)



smote_X,smote_y = smote.fit_sample(X,y)

columns=X.columns

smote_X = pd.DataFrame(data=smote_X,columns=columns)

smote_y= pd.DataFrame(data=smote_y,columns=['y'])



print("length of oversampled data is ",len(smote_X))

print("Number of no subscription in oversampled data",len(smote_y[smote_y['y']==0]))

print("Number of subscription",len(smote_y[smote_y['y']==1]))

print("Proportion of no subscription data in oversampled data is ",len(smote_y[smote_y['y']==0])/len(smote_X))

print("Proportion of subscription data in oversampled data is ",len(smote_y[smote_y['y']==1])/len(smote_X))

print(len(smote_X))
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=45,test_size=0.2)
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X_train,y_train)



pred = model.predict(X_test)

print('Accuracy score : ',accuracy_score(y_test,pred)*100,'%')

print('Confusion Matrix : ')

print(confusion_matrix(y_test,pred))



algo_dict = {}

algo_dict['LogisticRegression'] = accuracy_score(y_test,pred)
from sklearn.ensemble import RandomForestClassifier

forest_model = RandomForestClassifier().fit(X_train,y_train)
pred = forest_model.predict(X_test)

print('Accuracy score : ',accuracy_score(y_test,pred)*100,'%')

print('f1_score : ',f1_score(y_test,pred))

print('recall_score : ',recall_score(y_test,pred))

print('precision_score : ',precision_score(y_test,pred))

print('Confusion Matrix : ')

print(confusion_matrix(y_test,pred))

algo_dict['RandomForestClassifier'] = accuracy_score(y_test,pred)
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier().fit(X_train,y_train)



pred = model.predict(X_test)

print('Accuracy score : ',accuracy_score(y_test,pred)*100,'%')

print('f1_score : ',f1_score(y_test,pred))

print('recall_score : ',recall_score(y_test,pred))

print('precision_score : ',precision_score(y_test,pred))

print('Confusion Matrix : ')

print(confusion_matrix(y_test,pred))

algo_dict['GradientBoostingClassifier'] = accuracy_score(y_test,pred)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier().fit(X_train,y_train)



pred = model.predict(X_test)

print('Accuracy score : ',accuracy_score(y_test,pred)*100,'%')

print('f1_score : ',f1_score(y_test,pred))

print('recall_score : ',recall_score(y_test,pred))

print('precision_score : ',precision_score(y_test,pred))

print('Confusion Matrix : ')

print(confusion_matrix(y_test,pred))

algo_dict['DecisionTreeClassifier'] = accuracy_score(y_test,pred)
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB().fit(X_train,y_train)



pred = model.predict(X_test)

print('Accuracy score : ',accuracy_score(y_test,pred)*100,'%')

print('f1_score : ',f1_score(y_test,pred))

print('recall_score : ',recall_score(y_test,pred))

print('precision_score : ',precision_score(y_test,pred))

print('Confusion Matrix : ')

print(confusion_matrix(y_test,pred))

algo_dict['BernoulliNB'] = accuracy_score(y_test,pred)
from sklearn.svm import SVC

svc = SVC().fit(X_train,y_train)



pred = svc.predict(X_test)

print('Accuracy score : ',accuracy_score(y_test,pred)*100,'%')

print('Confusion Matrix : ')

print(confusion_matrix(y_test,pred))

algo_dict['SVC'] = accuracy_score(y_test,pred)
from sklearn.neighbors import KNeighborsClassifier

acc=[]

for i in range(3,25):

    model = KNeighborsClassifier(n_neighbors=i).fit(X_train,y_train)

    pred = model.predict(X_test)

    acc.append(accuracy_score(pred,y_test))

print('Max accuracy is :',max(acc))
sns.barplot(list(range(3,25)),acc,palette='muted')
sns.scatterplot(list(range(3,25)),acc,palette='muted')
for i in range(len(acc)):

    if acc[i]==max(acc):

        k=i+3

        break

print('Best k-value is : ',k)

model = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)

pred = model.predict(X_test)

print('\nKNearestClassifier model created successfully!')

print()

print('Accuracy score  : ',accuracy_score(pred,y_test))

print('Confusion Matrix : ')

print(confusion_matrix(y_test,pred))

algo_dict['KNeighborsClassifier'] = accuracy_score(y_test,pred)
from sklearn.linear_model import PassiveAggressiveClassifier

model = PassiveAggressiveClassifier().fit(X_train,y_train)



pred = model.predict(X_test)

print('Accuracy score : ',accuracy_score(y_test,pred)*100,'%')

print('Confusion Matrix : ')

print(confusion_matrix(y_test,pred))

algo_dict['PassiveAggressiveClassifier'] = accuracy_score(y_test,pred)
algo_dict
lst_1 = list(algo_dict.keys())

lst_2 = list(algo_dict.values())

fig,ax = plt.subplots(figsize=(16,5))

sns.barplot(lst_1,lst_2,ax=ax,palette='muted')

plt.show()
#Variable Importance for model with highest accuracy



imp_dict = { X.columns[i]:imp for i,imp in enumerate(forest_model.feature_importances_)}

sorted_imp_dict = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)

sorted_imp_dict
#Variable Importance Bar graph



lst_1 = list(imp_dict.keys())

lst_2 = list(imp_dict.values())

fig,ax = plt.subplots(figsize=(16,5))

with sns.color_palette('muted'):

    sns.barplot(y=lst_1,x=lst_2,ax=ax,palette='muted',orient="h")

plt.show()