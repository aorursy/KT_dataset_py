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
train = pd.read_csv("../input/sf-crime/train.csv.zip")
test = pd.read_csv("../input/sf-crime/test.csv.zip")
train
test
train.drop(["Resolution","Descript"],axis=1,inplace=True)
train
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train["Category"] = le.fit_transform(train["Category"])
train
train["DayOfWeek"].value_counts()

train["DayOfWeek"] = train["DayOfWeek"].map({'Monday':1,'Tuesday':2,'Wednesday':3,'Thrusday':4,'Friday':5,'Saturday':6,'Sunday':7})
test["DayOfWeek"] = test["DayOfWeek"].map({'Monday':1,'Tuesday':2,'Wednesday':3,'Thrusday':4,'Friday':5,'Saturday':6,'Sunday':7})
train
train = pd.get_dummies(train,columns=['PdDistrict'])
test = pd.get_dummies(test,columns=['PdDistrict'])
train
train["Dates"]
date_col = train["Dates"]
date_col2 = test["Dates"]
date_col = pd.to_datetime(date_col)
date_col2 = pd.to_datetime(date_col2)
train["year"]=date_col.dt.year
train["year"] = train["year"]-2000
test["year"]=date_col2.dt.year
test["year"] = test["year"]-2000
train["hour"]=date_col.dt.hour
train["month"]=date_col.dt.month
test["hour"]=date_col2.dt.hour
test["month"]=date_col2.dt.month
train["IsDay"] = date_col.dt.hour.apply(lambda h: 1 if (h>6 and h<19) else 0)
test["IsDay"] = date_col2.dt.hour.apply(lambda h: 1 if (h>6 and h<19) else 0)
train

test.info()
train.drop(["Dates"],axis=1,inplace=True)
test.drop(["Dates"],axis=1,inplace=True)
feature_col = ['DayOfWeek','X','Y','year','hour','month','IsDay','PdDistrict_BAYVIEW','PdDistrict_CENTRAL','PdDistrict_INGLESIDE','PdDistrict_MISSION','PdDistrict_NORTHERN','PdDistrict_PARK','PdDistrict_RICHMOND','PdDistrict_SOUTHERN','PdDistrict_TARAVAL','PdDistrict_TENDERLOIN']

train["DayOfWeek"].fillna(train["DayOfWeek"].mean(),inplace=True)
test["DayOfWeek"].fillna(train["DayOfWeek"].mean(),inplace=True)



train.info()


from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier()
clf1.fit(train[feature_col],train["Category"])
clf1.score(train[feature_col],train["Category"])
pred = clf1.predict(test[feature_col])
label = le.inverse_transform(pred)

label

df = pd.DataFrame(label)
df = pd.get_dummies(df)
l = [x for x in range(df.shape[0])]
l = np.asarray(l)
l = l.reshape((-1,1))


f = np.hstack((l,df))
f = pd.DataFrame(f)
f.columns = ['Id','ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY',
       'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE',
       'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
       'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD',
       'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS',
       'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES',
       'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE',
       'ROBBERY', 'RUNAWAY', 'SECONDARY CODES',
       'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE',
       'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC', 'TREA',
       'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS',
       'WEAPON LAWS']
f.to_csv("sub.csv",index=False)
#df = pd.get_dummies(df)
f


df
df
sub = pd.read_csv("../input/sf-crime/sampleSubmission.csv.zip")
sub
