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
train=pd.read_csv("/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv")

train.head()

#null column percentage



Col_With_NaN =train.isnull().sum() /len(train)*100

print(Col_With_NaN)
def MissingValue(train, Col_Dorp_thresold, Drop_raw_with_NaN = False):

    

    #for dropping column with NaN value

    Col_With_NaN =train.isnull().sum() /len(train)*100 #missing value percentage column vise

    Dropped_Columns = []

    for i in range(0,len(train.columns)):

        if Col_With_NaN[i] >= Col_Dorp_thresold:   #setting the threshold to drop columns%

            Dropped_Columns.append(train.columns[i])

    print(Dropped_Columns) #print the missing column name

    data = train.drop(Dropped_Columns,axis=1)

    

    #for dropping rows with NaN value

    if Drop_raw_with_NaN:

        Row_with_NaN =data.loc[pd.isnull(data).any(1), :].index

        print(Row_with_NaN)

        data = data.drop(Row_with_NaN,axis=0)

   

    return data



train=MissingValue(train,70,False) 

    
train.shape





train['employment_type']=train['employment_type'].fillna(-1)
train = train.drop(

        ['job_id', 'title', 'location', 'company_profile', 'description', 'requirements',

         'benefits'], axis=1).sort_index()
train.head()
numerical_features = ['telecommuting', 'has_company_logo', 'has_questions','employment_type']

label_features = [ 'required_experience', 'required_education', 'industry', 'function']
train['function']=train['function'].fillna(-1)

train['required_education']=train['required_education'].fillna(-1)

train['required_experience']=train['required_experience'].fillna(-1)

train['industry']=train['industry'].fillna(-1)
from sklearn.preprocessing import LabelEncoder

number= LabelEncoder()

train['function']=number.fit_transform(train['function'].astype('str'))

train['required_education']=number.fit_transform(train['required_education'].astype('str'))

train['required_experience']=number.fit_transform(train['required_experience'].astype('str'))

train['industry']=number.fit_transform(train['industry'].astype('str'))

train['employment_type']=number.fit_transform(train['employment_type'].astype('str'))

train['department']=number.fit_transform(train['department'].astype('str'))                                             

train.head()
X=train.drop("fraudulent", axis=1)

Y = train["fraudulent"]



X.head()

Y.head()



#X.shape[0]

#Y.shape[0]
from sklearn.svm import SVC, LinearSVC



svc = SVC()

svc.fit(X, Y)

Y_pred = svc.predict(train)

acc_svc = round(svc.score(X, Y) * 100, 2)

acc_svc
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X, Y)

Y_pred = logreg.predict(train)

acc_log = round(logreg.score(X, Y) * 100, 2)

acc_log