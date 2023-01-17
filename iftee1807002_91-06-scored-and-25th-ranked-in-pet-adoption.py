# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here are several helpful packages to load



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
train=pd.read_csv("../input/hackerearth-ml-challenge-pet-adoption/train.csv")

test=pd.read_csv("../input/hackerearth-ml-challenge-pet-adoption/test.csv")

train.head()
test.head()
train.shape,test.shape
train.info()
test.info()
train['breed_category'].value_counts()
#cheak

a=train['breed_category'][(np.isnan(train['condition']))]

a.value_counts()
#copy all test id to create submission file

test_id=test['pet_id']

#save the train left...it will use when the combine data will split into the previous train and test data after doing feature engineering

ntrain=train.shape[0]
#save target variable i.e label

y1=train['breed_category']

y2=train['pet_category']
#combine test and train data

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['breed_category','pet_category'], axis=1, inplace=True)
all_data['condition'].value_counts()
all_data['condition'].fillna(-1,inplace=True)
all_data['condition'].value_counts()
all_data.info()
all_data['issue_date']=pd.to_datetime(all_data['issue_date'])

all_data['listing_date']=pd.to_datetime(all_data['listing_date'])

x=[]

for d in all_data['issue_date']:

    y=d.month

    x.append(y)

all_data['issue_month']=x
x=[]

for d in all_data['listing_date']:

    y=d.month

    x.append(y)

all_data['listing_month']=x
x=[]

for d in all_data['listing_date']:

    y=d.year+(d.month/12.0)+(d.day/365.0)

    x.append(y)

all_data['modified_listing_date']=x
x=[]

for d in all_data['issue_date']:

    y=d.year+(d.month/12.0)+(d.day/365.0)

    x.append(y)

all_data['modified_issue_date']=x
all_data['took_time']=abs(all_data['modified_listing_date']-all_data['modified_issue_date'])


all_data['1stnum'] = all_data['pet_id'].str[:6]

all_data['1st2num'] = all_data['pet_id'].str[:7]
train = all_data[:ntrain]

test = all_data[ntrain:]
#drop some unnecessary features

x=train.drop(['pet_id','issue_date','listing_date','modified_issue_date'],axis=1)

test=test.drop(['pet_id','issue_date','listing_date','modified_issue_date'],axis=1)

x.select_dtypes(exclude='number').columns.to_list()
x.shape
x=pd.get_dummies(x)

test=pd.get_dummies(test)

x.shape,test.shape
a=set(x.columns)-set(test.columns)
a=list(a)

a
x=x.drop(a,axis=1)
x.shape,test.shape
#again combining

all_data = pd.concat((x, test)).reset_index(drop=True)
from sklearn import preprocessing

# Get column names first

names = all_data.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(all_data)

all_data = pd.DataFrame(scaled_df, columns=names)
x = all_data[:ntrain]

test = all_data[ntrain:]
from sklearn.model_selection import train_test_split

x1_train,x1_test,y1_train,y1_test=train_test_split(x,y2,test_size=0.2,random_state=44,shuffle=True)
from xgboost import XGBClassifier
model1 = XGBClassifier()

model1.fit(x1_train, y1_train)
#new_feat is new feature i.e the predicted pet_category of model 1 for train data

new_feat=model1.predict(x)

#output1 is new first output i.e the predicted pet_category of model 1 for test data

output1=model1.predict(test)

#vld1 is validation 1 i.e we'll check score with the predicted result of validation data of model 1

vld1=model1.predict(x1_test)
x2 = pd.DataFrame(x, columns=names)

test2 = pd.DataFrame(test, columns=names)
#the predicted pet_category of model 1 for train data is used as a input variable or feature of the train data of model 2

x2['output1']=new_feat

#the predicted pet_category of model 1 for test data is used as a input variable or feature of the test data of model 2

test2['output1']=output1
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y1,test_size=0.2,random_state=44)
model2 = XGBClassifier()

model2.fit(x2_train, y2_train)
#output 2 is the predicted breed_category of model 2 for test data

output2=model2.predict(test)

#vld2 is validation 2 i.e we'll check score with the predicted result of validation data of model 2

vld2=model2.predict(x2_test)
from sklearn.metrics  import f1_score
s1=f1_score(y1_test,vld1,average='weighted')

s2=f1_score(y2_test,vld2,average='weighted')

accuracy=100*((s1+s2)/2)

accuracy
sub_new=pd.DataFrame({

    "pet_id":test_id,

    "breed_category":output2,

    "pet_category":output1

})

sub_new.to_csv("sub_new.csv",index=False)
y1.value_counts()
y2.value_counts()
from scipy.stats import skew
y1.skew(axis = 0, skipna = True),y2.skew(axis = 0, skipna = True)