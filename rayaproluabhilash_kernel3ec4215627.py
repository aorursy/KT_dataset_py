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
df=pd.read_csv("/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv")
df.head()
df.tail()
df.isnull().sum()
df.location.value_counts()
df.company_profile.value_counts()
df.description.value_counts()
df.requirements.value_counts()
df.employment_type.value_counts()
df.industry.value_counts()
df.job_id.value_counts()
df.title.value_counts()
df.salary_range.value_counts()
df.benefits.value_counts()
df.has_company_logo.value_counts()
df.required_experience.value_counts()
df.function.value_counts()
df.location.fillna("GB, LND, London",inplace=True)

df.company_profile.fillna("We help teachers get safe &amp; secure jobs abroad :)",inplace=True)

df.description.fillna("Play with kids, get paid for it Love travel? Jobs in Asia$1,500+ USD monthly ($200 Cost of living)Housing provided (Private/Furnished)Airfare ReimbursedExcellent for student loans/credit cardsGabriel Adkins",inplace=True)

df.requirements.fillna("University degree required. TEFL / TESOL / CELTA or teaching experience preferred but not necessaryCanada/US passport holders only",inplace=True)

df.employment_type.fillna("Full-time",inplace=True)

df.industry.fillna("Information Technology and Services",inplace=True)

df.job_id.fillna("14994",inplace=True)

df.title.fillna("English Teacher Abroad",inplace=True)

df.salary_range.fillna("0-0",inplace=True)

df.benefits.fillna("See job description",inplace=True)

df.has_company_logo.fillna("1",inplace=True)

df.required_experience.fillna("Mid-Senior level",inplace=True)

df.required_education.fillna("Information Technology",inplace=True)

df.function.fillna("Information Technology",inplace=True)

df.department.fillna("Information Technology",inplace=True)



df.isnull().sum()
df=df.drop(['department','has_questions','benefits','salary_range','telecommuting'],axis=1)

df.info()
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

df['title']=label.fit_transform(df['title'])

df['requirements']=label.fit_transform(df['requirements'])

df['industry']=label.fit_transform(df['industry'])

df['function']=label.fit_transform(df['function'])

df['required_experience']=label.fit_transform(df['required_experience'])

df['required_education']=label.fit_transform(df['required_education'])

df['description']=label.fit_transform(df['description'])

df['location']=label.fit_transform(df['location'])

df['employment_type']=label.fit_transform(df['employment_type'])

df['company_profile']=label.fit_transform(df['company_profile'])
df.info()
df.fraudulent.value_counts()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot('fraudulent', data=df)

plt.title('Fraudulent Classes', fontsize=14)

plt.ylabel("Frequency")

plt.show()
from sklearn.model_selection import train_test_split

train, test=train_test_split(df, test_size=0.2, random_state=1)
def data_splitting(df):

    x=df.drop(['fraudulent'], axis=1)

    y=df['fraudulent']

    return x, y



x_train, y_train=data_splitting(train)

x_test, y_test=data_splitting(test)
from sklearn.utils import resample

import imblearn

from imblearn.over_sampling import SMOTE

sm = SMOTE()

x_train, y_train = sm.fit_sample(x_train, y_train)
x_train = pd.DataFrame(data=x_train)

x_train.columns = ['job_id','title','location','company_profile','description','requirements','has_company_logo','employment_type','required_experience','required_education','industry','function'] 

y_train = pd.DataFrame(data = y_train)

y_train.columns = ['fraudulent']
sns.countplot('fraudulent', data=y_train)

plt.title('Fraudulent Classes', fontsize=14)

plt.ylabel("Frequency")

plt.show()
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



log_model=LogisticRegression()

log_model.fit(x_train, y_train)

prediction=log_model.predict(x_test)

score= accuracy_score(y_test, prediction)

print(score)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

smote = XGBClassifier()

smote.fit(x_train, y_train)



# Predict on test

smote_pred = smote.predict(x_test)

accuracy = accuracy_score(y_test, smote_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()

reg.fit(x_train , y_train)

reg_train = reg.score(x_train , y_train)

reg_test = reg.score(x_test , y_test)





print(reg_train)

print(reg_test)