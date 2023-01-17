import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train=pd.read_csv('/kaggle/input/titanic/train.csv')



df_test=pd.read_csv('/kaggle/input/titanic/test.csv')
df_all=pd.concat([df_train,df_test],ignore_index=True)

df_all.head()
plt.figure(figsize=(16,5))

plt.suptitle('Percentage of survivors with of kids younger than 16',fontsize=16)

plt.subplot2grid((1,3),(0,0))

df_train.Survived[(df_train["Age"]<16) & (df_train["Sex"]=='male')].value_counts(normalize=True,sort=False).plot(kind="bar",alpha=0.5)

plt.title('Percentage of Boy survivors')

plt.subplot2grid((1,3),(0,1))

df_train.Survived[(df_train["Age"]<16) & (df_train["Sex"]=='female')].value_counts(normalize=True,sort=False).plot(kind="bar",alpha=0.5)

plt.title('Percentage of Girl survivors')

plt.subplot2grid((1,3),(0,2))

df_train.Survived[(df_train["Age"]<16) ].value_counts(normalize=True,sort=False).plot(kind="bar",alpha=0.5)

plt.title('Percentage of Kid survivors')

plt.show()
plt.figure(figsize=(16,5))

plt.suptitle('Percentage of survivors with of kids younger than 17',fontsize=16)

plt.subplot2grid((1,3),(0,0))

df_train.Survived[(df_train["Age"]<17) & (df_train["Sex"]=='male')].value_counts(normalize=True,sort=False).plot(kind="bar",alpha=0.5)

plt.title('Percentage of Boy survivors')

plt.subplot2grid((1,3),(0,1))

df_train.Survived[(df_train["Age"]<17) & (df_train["Sex"]=='female')].value_counts(normalize=True,sort=False).plot(kind="bar",alpha=0.5)

plt.title('Percentage of Girl survivors')

plt.subplot2grid((1,3),(0,2))

df_train.Survived[(df_train["Age"]<17) ].value_counts(normalize=True,sort=False).plot(kind="bar",alpha=0.5)

plt.title('Percentage of Kid survivors')

plt.show()
girl_age=df_all.Age[(df_all.Sex=='female') & (df_all.Age<16)].median(skipna=True)

boy_age=df_all.Age[(df_all.Sex=='male') & (df_all.Age<16)].median(skipna=True)

man_age=df_all.Age[(df_all.Sex=='male') & (df_all.Age>=16)].median(skipna=True)

woman_age=df_all.Age[(df_all.Sex=='female') & (df_all.Age>=16)].median(skipna=True)



print('Median Age of minor girls',girl_age)

print('Median Age of minor boys',boy_age)

print('Median Age of adult men',man_age)

print('Median Age of adult women',woman_age)
fare_girl=df_all.Fare[(df_all.Sex=='female') & (df_all.Age<16)].median(skipna=True)

fare_boy=df_all.Fare[(df_all.Sex=='male') & (df_all.Age<16)].median(skipna=True)

fare_man=df_all.Fare[(df_all.Sex=='male') & (df_all.Age>=16)].median(skipna=True)

fare_woman=df_all.Fare[(df_all.Sex=='female') & (df_all.Age>=16)].median(skipna=True)



print('Median fare of minor girls',fare_girl)

print('Median fare of minor boys',fare_boy)

print('Median fare of adult men',fare_man)

print('Median fare of adult women',fare_woman)
# Extracting passenger titles from Training set

names=df_train.Name.str.split(', ',expand=True)

df_train["LastName"]=names[0]

title_names=names[1].str.split('.',expand=True)

df_train["Title"]=title_names[0]

df_train["Title"]=df_train["Title"].astype(str)

df_train.drop(columns=["LastName","Name"],inplace=True)
# Extracting passenger titles from testing set

names=df_test.Name.str.split(', ',expand=True)

df_test["LastName"]=names[0]

title_names=names[1].str.split('.',expand=True)

df_test["Title"]=title_names[0]

df_train["Title"]=df_train["Title"].astype(str)

df_test.drop(columns=["LastName","Name"],inplace=True)
import warnings

warnings.filterwarnings('ignore')



men_title=['Mr', 'Don', 'Rev', 'Dr', 'Mme','Major', 'Sir', 'Mlle', 'Col', 'Capt','Jonkheer']

df_train["Age"][(df_train["Sex"]=='male') & (df_train["Title"].isin(['Master'])) & df_train["Age"].isna()]=boy_age

df_train["Age"][(df_train["Sex"]=='male') & (df_train["Title"].isin(men_title)) & df_train["Age"].isna()]=man_age

df_train["Age"][(df_train["Sex"]=='female') & (df_train["Title"].isin(['Mrs'])) & df_train["Age"].isna()]=woman_age

df_train["Age"][(df_train["Age"].isna()) & (df_train["Parch"]+df_train["SibSp"]>=1)]=girl_age

df_train["Age"][(df_train["Age"].isna())]=woman_age





df_train["Fare"][(df_train["Sex"]=='female') & (df_train["Age"]<16)]=fare_girl

df_train["Fare"][(df_train["Sex"]=='male') & (df_train["Age"]<16)]=fare_boy

df_train["Fare"][(df_train["Sex"]=='male')]=fare_man

df_train["Fare"][(df_train["Sex"]=='female')]=fare_woman



df_train["Embarked"] = df_train["Embarked"].fillna("S")
import warnings

warnings.filterwarnings('ignore')



men_title=['Mr', 'Don', 'Rev', 'Dr', 'Mme','Major', 'Sir', 'Mlle', 'Col', 'Capt','Jonkheer']

df_test["Age"][(df_test["Sex"]=='male') & (df_test["Title"].isin(['Master'])) & df_test["Age"].isna()]=boy_age

df_test["Age"][(df_test["Sex"]=='male') & (df_test["Title"].isin(men_title)) & df_test["Age"].isna()]=man_age

df_test["Age"][(df_test["Sex"]=='female') & (df_test["Title"].isin(['Mrs'])) & df_test["Age"].isna()]=woman_age

df_test["Age"][(df_test["Age"].isna()) & (df_test["Parch"]+df_test["SibSp"]>=1)]=girl_age

df_test["Age"][(df_test["Age"].isna())]=woman_age





df_test["Fare"][(df_test["Sex"]=='female') & (df_test["Age"]<16)]=fare_girl

df_test["Fare"][(df_test["Sex"]=='male') & (df_test["Age"]<16)]=fare_boy

df_test["Fare"][(df_test["Sex"]=='male')]=fare_man

df_test["Fare"][(df_test["Sex"]=='female')]=fare_woman



df_test["Embarked"] = df_test["Embarked"].fillna("S")
# train=df_train.copy()

# test=df_test.copy()

df_train["Family"]=df_train["SibSp"]+df_train["Parch"]

df_test["Family"]=df_test["SibSp"]+df_test["Parch"]

# train.drop(columns=['Ticket','Cabin','Title','SibSp','Parch'],inplace=True)

# test.drop(columns=['Ticket','Cabin','Title','SibSp','Parch'],inplace=True)
train_dum = pd.get_dummies(df_train, columns=["Sex",'Embarked','Pclass'], prefix=["Sex_is","Emb_is",'Class_is'] )

train_dum
test_dum = pd.get_dummies(df_test, columns=["Sex",'Embarked','Pclass'], prefix=["Sex_is","Emb_is",'Class_is'] )

test_dum
cols=['Age','Fare','Family','Sex_is_female','Sex_is_male','Emb_is_C','Emb_is_Q','Emb_is_S','Class_is_1',

      'Class_is_2','Class_is_3']

features=train_dum[cols].values

target=train_dum['Survived'].values



test_features=test_dum[cols].values
from sklearn.linear_model import LogisticRegressionCV

log_reg_cv=LogisticRegressionCV(cv=5, random_state=0)

log_reg_cv.fit(features,target)

log_reg_cv.score(features,target)
hyp=log_reg_cv.predict(test_features)
from sklearn import preprocessing

poly=preprocessing.PolynomialFeatures(degree=4,interaction_only=True)

poly_features=poly.fit_transform(features)
log_reg_cv.fit(poly_features,target)

log_reg_cv.score(poly_features,target)
test_poly_features=poly.fit_transform(test_features)

hyp_poly=log_reg_cv.predict(test_poly_features)

# 1-sum((hyp_poly-true_results["Survived"].values)**2)/len(hyp_poly)                            
PassID_test=df_test["PassengerId"].values.astype(int)

poly_LR_prediction = pd.DataFrame(hyp_poly, PassID_test, columns = ["Survived"])



poly_LR_prediction.to_csv('poly_LR_submission.csv', index_label = ["PassengerId"])